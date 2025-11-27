# Troubleshooting: "Failed to fetch" Error on cloudvina.in

## Problem
- ✅ **cloudvina-3n3v.vercel.app** → Login works
- ❌ **cloudvina.in** → "Failed to fetch" error on login

## Root Cause
The custom domain `cloudvina.in` is missing the `VITE_API_URL` environment variable in Vercel, causing the frontend to fail connecting to the backend.

---

## Solution: Set Environment Variables in Vercel

### Step 1: Go to Vercel Dashboard
1. Open [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your **CloudVina** project
3. Click **Settings** → **Environment Variables**

### Step 2: Add Required Variable
Add the following environment variable:

| Name | Value | Environments |
|------|-------|--------------|
| `VITE_API_URL` | `https://cloudvina-api.onrender.com` | Production, Preview |

### Step 3: Redeploy
1. Go to **Deployments** tab
2. Click the **3-dot menu** on the latest deployment
3. Select **"Redeploy"**
4. Check **"Use existing Build Cache"** → **NO** (force fresh build)
5. Click **"Redeploy"**

---

## Verification Steps

### 1. Check API Connection
Open browser console (F12) on `cloudvina.in` and run:
```javascript
console.log(import.meta.env.VITE_API_URL)
```
Should show: `https://cloudvina-api.onrender.com`

### 2. Test API Directly
```bash
curl https://cloudvina-api.onrender.com/health
```
Should return: `{"status":"healthy",...}`

### 3. Check CORS
The backend at `api/main.py` already includes `https://cloudvina.in` in CORS:
```python
allow_origins=[
    "https://cloudvina.in",
    "https://www.cloudvina.in",
    ...
]
```

---

## Alternative: Quick Fix (If Above Doesn't Work)

If environment variables don't propagate, you can temporarily hardcode the API URL:

**File:** `web/src/config.js`
```javascript
// Remove the env variable check, use hardcoded URL
export const API_URL = 'https://cloudvina-api.onrender.com'
```

Then commit and push:
```bash
git add web/src/config.js
git commit -m "Fix: Hardcode API URL for production"
git push
```

---

## Why This Happens

Vercel deployments can have **different environment variables** for:
- **Production** (cloudvina.in)
- **Preview** (cloudvina-3n3v.vercel.app)

The preview URL might have inherited the correct variables from initial setup, but the custom domain assignment didn't copy them over.

---

## Prevention

Always verify environment variables are set for **ALL environments** in Vercel:
- ✅ Production
- ✅ Preview
- ✅ Development
