# Secure Login Setup: Customer Financial Behaviour Dashboard

This upgrade adds the following protected account flow:

1. **Register account** using email and a strong password.
2. Firebase sends an **email-verification link**.
3. The dashboard stays locked until the user verifies the email address.
4. **Sign in** creates a Firebase ID-token session, refreshed automatically before expiry.
5. **Password reset** sends a Firebase reset link without revealing whether the email is registered.
6. **Log out** clears the Streamlit session.

Passwords are never saved in your Python files, CSV files, or a local database. Firebase Authentication manages password storage and reset links.

## 1. Create and configure Firebase

1. Open the Firebase Console and create a project, for example `customer-financial-behaviour`.
2. In **Authentication** > **Sign-in method**, enable **Email/Password**.
3. In **Authentication** > **Settings**, configure a Firebase password policy as an extra server-side security control.
4. In **Project settings** > **General**, copy the project's **Web API Key**.

## 2. Add your API key safely

1. Open the `.streamlit` folder in this project.
2. Copy `secrets.toml.template` and rename the copy to `secrets.toml`.
3. Replace the placeholder:

```toml
FIREBASE_API_KEY = "PASTE_YOUR_FIREBASE_WEB_API_KEY_HERE"
```

The real `secrets.toml` is ignored by Git through `.gitignore`.

## 3. Install packages and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep your existing `data/origination_data-2.csv` file in the `data` folder if you want the dashboard to load its default dataset without an upload.

## 4. Test checklist

1. Register with a new email and strong password.
2. Confirm that the dashboard does **not** open yet.
3. Open the Firebase verification email and select its verification link.
4. Return to Streamlit and select **I have verified my email**.
5. Log out, then sign in again.
6. Test **Password reset** using the same email.
7. Confirm that an unknown email receives the same generic reset-screen message.

## Important deployment notes

- Do not use the old `.gitignore` file containing `*`; it prevents normal project files from being committed.
- Do not commit `.streamlit/secrets.toml`.
- Use a separate Firebase project/API key for development and production.
- Firebase Auth uses the configured email templates for registration verification and password-reset messages.
