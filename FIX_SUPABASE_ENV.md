# URGENT FIX: Missing Supabase Environment Variables

## Problem Identified ✅
The error console shows:
```
ERR_NAME_NOT_RESOLVED: https://placeholder.supabase.co/auth/v1/token
```

This means `cloudvina.in` is using **placeholder** Supabase credentials instead of your **real** ones!

---

## Required Environment Variables

You need to add **THREE** environment variables to Vercel Production:

### 1. VITE_SUPABASE_URL
Your actual Supabase project URL (looks like: `https://xxxxx.supabase.co`)

### 2. VITE_SUPABASE_ANON_KEY
Your Supabase anonymous/public key (long string starting with `eyJ...`)

### 3. VITE_API_URL
`https://cloudvina-api.onrender.com` ✅ (You already added this)

---

## How to Find Your Supabase Credentials

### Option 1: From Supabase Dashboard
1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your **CloudVina** project
3. Go to **Settings** → **API**
4. Copy:
   - **Project URL** → This is `VITE_SUPABASE_URL`
   - **Project API keys** → Copy the `anon` `public` key → This is `VITE_SUPABASE_ANON_KEY`

### Option 2: From Your `.env` File (Local Development)
Check if you have a `.env` file in your `web/` folder:
```bash
cat web/.env
```

Look for lines like:
```
VITE_SUPABASE_URL=https://xxxxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Steps to Fix

### 1. Add Variables to Vercel
1. [Vercel Dashboard](https://vercel.com/dashboard) → CloudVina project
2. **Settings** → **Environment Variables**
3. Click **Add New**
4. Add each variable:

   **Variable 1:**
   - Key: `VITE_SUPABASE_URL`
   - Value: `https://your-project-id.supabase.co`
   - Environments: ✅ Production ✅ Preview ✅ Development

   **Variable 2:**
   - Key: `VITE_SUPABASE_ANON_KEY`
   - Value: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (your full key)
   - Environments: ✅ Production ✅ Preview ✅ Development

   **Variable 3:** (Already done)
   - Key: `VITE_API_URL`
   - Value: `https://cloudvina-api.onrender.com`
   - Environments: ✅ Production ✅ Preview ✅ Development

5. Click **Save**

### 2. Redeploy
1. **Deployments** tab
2. Latest deployment → **⋮ (3 dots)** → **Redeploy**
3. ❌ **Uncheck** "Use existing Build Cache"
4. Click **Redeploy**

### 3. Wait & Test
- Wait **2-3 minutes** for build to complete
- Go to `https://cloudvina.in/login`
- Press `Ctrl + Shift + R` to hard refresh
- Try logging in → Should work! ✅

---

## Verification

After redeploying, open browser console (F12) on `cloudvina.in` and check:

```javascript
// Should NOT show "placeholder.supabase.co"
console.log(import.meta.env.VITE_SUPABASE_URL)
// Should show your actual Supabase URL
```

---

## Why This Happened

Your `web/src/supabaseClient.js` has fallback values:
```javascript
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://placeholder.supabase.co'
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'placeholder-key'
```

When Vercel builds **without** these environment variables, it uses the placeholders, which obviously don't work!
