"""Streamlit UI for registration, email verification, login and password reset."""

from __future__ import annotations

import re
import time
from typing import Any

import streamlit as st

from utils.auth import (
    AuthConfigurationError,
    AuthError,
    create_account,
    firebase_error_message,
    get_account,
    get_firebase_api_key,
    refresh_session_if_needed,
    request_password_reset,
    send_verification_email,
    sign_in,
)

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MINIMUM_PASSWORD_LENGTH = 10
MAXIMUM_PASSWORD_LENGTH = 128
AUTH_CHECK_INTERVAL_SECONDS = 300


def initialise_auth_state() -> None:
    """Create predictable defaults once for every browser session."""
    defaults: dict[str, Any] = {
        "authenticated": False,
        "auth_session": None,
        "auth_user": None,
        "auth_pending_verification": None,
        "auth_last_check": 0.0,
        "auth_login_failures": 0,
        "auth_locked_until": 0.0,
        "auth_notice": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def logout_user(notice: str = "") -> None:
    """Delete authentication data from the current Streamlit browser session."""
    for key in (
        "authenticated",
        "auth_session",
        "auth_user",
        "auth_pending_verification",
        "auth_last_check",
        "auth_login_failures",
        "auth_locked_until",
        "auth_notice",
    ):
        st.session_state.pop(key, None)

    initialise_auth_state()
    if notice:
        st.session_state["auth_notice"] = notice


def require_verified_user() -> dict[str, Any] | None:
    """Return a verified Firebase user, or clear invalid/expired dashboard access."""
    initialise_auth_state()

    if not st.session_state.get("authenticated"):
        return None

    session = st.session_state.get("auth_session")
    if not isinstance(session, dict):
        logout_user("Your sign-in session ended. Please sign in again.")
        return None

    try:
        session = refresh_session_if_needed(session)
        st.session_state["auth_session"] = session

        user = st.session_state.get("auth_user")
        must_check_account = (
            not isinstance(user, dict)
            or (time.time() - float(st.session_state.get("auth_last_check") or 0))
            >= AUTH_CHECK_INTERVAL_SECONDS
        )

        if must_check_account:
            user = get_account(session["id_token"])
            st.session_state["auth_user"] = user
            st.session_state["auth_last_check"] = time.time()

        if user.get("disabled", False):
            logout_user("This account is unavailable. Contact the dashboard administrator.")
            return None

        if not user.get("email_verified", False):
            st.session_state["auth_pending_verification"] = session
            st.session_state["auth_notice"] = "Verify your email address before opening the dashboard."
            st.session_state["authenticated"] = False
            st.session_state["auth_session"] = None
            st.session_state["auth_user"] = None
            return None

        return user

    except AuthError:
        logout_user("Your sign-in session ended. Please sign in again.")
        return None


def _is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(email.strip()))


def _password_issues(password: str) -> list[str]:
    issues: list[str] = []

    if len(password) < MINIMUM_PASSWORD_LENGTH:
        issues.append(f"at least {MINIMUM_PASSWORD_LENGTH} characters")
    if len(password) > MAXIMUM_PASSWORD_LENGTH:
        issues.append(f"no more than {MAXIMUM_PASSWORD_LENGTH} characters")
    if not re.search(r"[A-Z]", password):
        issues.append("one uppercase letter")
    if not re.search(r"[a-z]", password):
        issues.append("one lowercase letter")
    if not re.search(r"\d", password):
        issues.append("one number")
    if not re.search(r"[^A-Za-z0-9]", password):
        issues.append("one symbol")

    return issues


def _record_failed_login() -> None:
    failures = int(st.session_state.get("auth_login_failures") or 0) + 1
    st.session_state["auth_login_failures"] = failures

    if failures >= 5:
        st.session_state["auth_locked_until"] = time.time() + 60
        st.session_state["auth_login_failures"] = 0


def _login_is_temporarily_locked() -> bool:
    locked_until = float(st.session_state.get("auth_locked_until") or 0)
    return time.time() < locked_until


def _remaining_lock_seconds() -> int:
    locked_until = float(st.session_state.get("auth_locked_until") or 0)
    return max(0, int(locked_until - time.time()))


def _complete_login(session: dict[str, Any], user: dict[str, Any]) -> None:
    st.session_state["authenticated"] = True
    st.session_state["auth_session"] = session
    st.session_state["auth_user"] = user
    st.session_state["auth_last_check"] = time.time()
    st.session_state["auth_pending_verification"] = None
    st.session_state["auth_login_failures"] = 0
    st.session_state["auth_locked_until"] = 0.0
    st.session_state["auth_notice"] = ""
    st.rerun()


def _render_pending_verification() -> None:
    pending = st.session_state.get("auth_pending_verification")
    if not isinstance(pending, dict):
        return

    email = str(pending.get("email") or "your registered email address")

    st.subheader("Verify your email address")
    st.info(
        f"A verification link was sent to **{email}**. Open the link in your email, "
        "then return here and select **I have verified my email**."
    )

    verify_col, resend_col, back_col = st.columns(3)

    with verify_col:
        verified_now = st.button(
            "I have verified my email",
            use_container_width=True,
            key="check_email_verification",
        )
    with resend_col:
        resend = st.button(
            "Resend verification email",
            use_container_width=True,
            key="resend_email_verification",
        )
    with back_col:
        back_to_login = st.button(
            "Back to sign in",
            use_container_width=True,
            key="back_to_login_from_verification",
        )

    if back_to_login:
        st.session_state["auth_pending_verification"] = None
        st.rerun()

    if verified_now:
        try:
            updated_session = refresh_session_if_needed(pending)
            account = get_account(updated_session["id_token"])

            if account.get("email_verified", False):
                _complete_login(updated_session, account)
            else:
                st.warning("Firebase still shows this email as unverified. Open the newest verification email and try again.")
        except AuthError as exc:
            st.error(firebase_error_message(exc, context="verification"))

    if resend:
        try:
            updated_session = refresh_session_if_needed(pending)
            send_verification_email(updated_session["id_token"])
            st.session_state["auth_pending_verification"] = updated_session
            st.success("A new verification email has been sent.")
        except AuthError as exc:
            st.error(firebase_error_message(exc, context="verification"))


def _render_login_tab() -> None:
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email address", key="login_email", placeholder="name@example.com")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if not submitted:
        return

    if _login_is_temporarily_locked():
        st.error(f"Too many local sign-in attempts. Try again in {_remaining_lock_seconds()} seconds.")
        return

    email = email.strip().lower()
    if not _is_valid_email(email) or not password:
        st.error("Enter a valid email address and password.")
        return

    try:
        session = sign_in(email, password)
        account = get_account(session["id_token"])

        if not account.get("email_verified", False):
            st.session_state["auth_pending_verification"] = session
            st.warning("Your account exists, but its email address is not verified yet.")
            return

        _complete_login(session, account)

    except AuthError as exc:
        if exc.code in {
            "EMAIL_NOT_FOUND",
            "INVALID_PASSWORD",
            "INVALID_LOGIN_CREDENTIALS",
            "INVALID_EMAIL",
        }:
            _record_failed_login()
        st.error(firebase_error_message(exc, context="login"))


def _render_register_tab() -> None:
    st.caption(
        "Password requirement: at least 10 characters with uppercase, lowercase, number and symbol."
    )

    with st.form("register_form", clear_on_submit=False):
        email = st.text_input("Email address", key="register_email", placeholder="name@example.com")
        password = st.text_input("Create password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm password", type="password", key="register_confirm_password")
        submitted = st.form_submit_button("Create account", use_container_width=True)

    if not submitted:
        return

    email = email.strip().lower()
    issues = _password_issues(password)

    if not _is_valid_email(email):
        st.error("Enter a valid email address.")
        return
    if password != confirm_password:
        st.error("The password and confirmation password do not match.")
        return
    if issues:
        st.error("Password must contain " + ", ".join(issues) + ".")
        return

    try:
        session, verification_error = create_account(email, password)
        st.session_state["auth_pending_verification"] = session

        if verification_error is None:
            st.success("Account created. Check your inbox for the verification link.")
        else:
            st.warning(
                "Your account was created, but the verification email could not be sent yet. "
                "Use Resend verification email below."
            )
        st.rerun()

    except AuthError as exc:
        st.error(firebase_error_message(exc, context="register"))


def _render_password_reset_tab() -> None:
    st.write("Enter your email and Firebase will send a secure password-reset link when an account is available.")

    with st.form("password_reset_form", clear_on_submit=True):
        email = st.text_input("Email address", key="reset_email", placeholder="name@example.com")
        submitted = st.form_submit_button("Send password reset link", use_container_width=True)

    if not submitted:
        return

    email = email.strip().lower()
    if not _is_valid_email(email):
        st.error("Enter a valid email address.")
        return

    try:
        request_password_reset(email)
        # Deliberately generic: does not expose whether this email is registered.
        st.success("If an account exists for this email address, a password-reset link has been sent.")
    except AuthError as exc:
        st.error(firebase_error_message(exc, context="reset"))


def show_auth_page() -> None:
    """Display the complete unauthenticated account area."""
    initialise_auth_state()

    try:
        get_firebase_api_key()
    except AuthConfigurationError:
        st.error("Authentication needs Firebase configuration before it can run.")
        st.code(
            'FIREBASE_API_KEY = "YOUR_FIREBASE_WEB_API_KEY"',
            language="toml",
        )
        st.caption("Save this in .streamlit/secrets.toml, then restart Streamlit.")
        return

    st.markdown(
        """
        <style>
            .auth-hero {
                max-width: 760px;
                margin: 1rem auto 0.8rem auto;
                padding: 2rem;
                border-radius: 24px;
                background: rgba(15, 23, 42, 0.62);
                border: 1px solid rgba(255, 255, 255, 0.20);
                backdrop-filter: blur(18px);
            }
            .auth-hero p {
                opacity: 0.88;
                margin-bottom: 0;
            }
        </style>
        <div class="auth-hero">
            <h1>Customer Financial Behaviour</h1>
            <p>Secure dashboard access with verified email accounts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    notice = str(st.session_state.get("auth_notice") or "")
    if notice:
        st.warning(notice)
        st.session_state["auth_notice"] = ""

    auth_container = st.container(border=True)
    with auth_container:
        if st.session_state.get("auth_pending_verification"):
            _render_pending_verification()
            return

        login_tab, register_tab, reset_tab = st.tabs(
            ["Sign in", "Register account", "Password reset"]
        )

        with login_tab:
            _render_login_tab()
        with register_tab:
            _render_register_tab()
        with reset_tab:
            _render_password_reset_tab()
