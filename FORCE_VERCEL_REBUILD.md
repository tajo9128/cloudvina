# Force Vercel to Use Environment Variables

## Problem
Environment variables are set in Vercel, but the deployed site still uses placeholder values (`placeholder.supabase.co`).

## Solution: Delete & Redeploy Production Domain

### Step 1: Check Current Deployment Assignment
1. Vercel Dashboard → BioDockify project → **Settings** → **Domains**
2. Find `BioDockify.in` in the list
3. Note which **Git Branch** it's assigned to (probably `main`)
4. Note which **Production Deployment** it's pointing to

### Step 2: Force Fresh Production Build

**Option A: Via Vercel CLI (Fastest)**
```bash
# Install Vercel CLI if not already installed
npm i -g vercel

# Login
vercel login

# Navigate to your web directory
cd web

# Deploy to production, forcing rebuild
vercel --prod --force
```

**Option B: Via Dashboard (Manual)**
1. Go to **Deployments** tab
2. Find the latest successful deployment
3. Click **⋮ (3 dots)** → **Redeploy**
4. **CRITICAL:** Uncheck "Use existing Build Cache" ❌
5. Ensure it says "Production" not "Preview"
6. Click **Redeploy**

### Step 3: Verify Environment Variables Were Used

After deployment completes:
1. Go to the deployment logs
2. Search for "Building..." section
3. You should see:
   ```
   ✓ Environment variables loaded
   ```
4. The build should **NOT** show warnings about missing `VITE_SUPABASE_URL`

### Step 4: Clear Domain Assignment (If Still Not Working)

If redeploying doesn't work:
1. **Settings** → **Domains**
2. Find `BioDockify.in` → Click **Edit**
3. Temporarily **Remove** the domain assignment
4. Click **Save**
5. Force a new production deployment (Option A or B above)
6. **Re-add** `BioDockify.in` as the production domain
7. This forces Vercel to route the domain to the fresh build

---

## Verification Commands

### Check Environment Variables in Deployment
```bash
# SSH into a preview deployment (if possible) - not available on hobby plan
# OR check the build logs for environment variable loading
```

### Test API from Command Line
```bash
# This should return 200 OK
curl -I https://BioDockify-api.onrender.com/health
```

### Browser Console Test (After Redeploy)
```javascript
// Open console on BioDockify.in
console.log(import.meta.env.VITE_SUPABASE_URL)
// Should NOT show "placeholder.supabase.co"
// Should show your actual Supabase URL
```

---

## Why This Happens

Vercel's build cache can sometimes "lock in" missing environment variables. Even when you add them later, cached builds continue using the old (placeholder) values.

The `--force` flag or unchecking "Use existing Build Cache" ensures Vite rebuilds from scratch and pulls fresh environment variables.

---

## Alternative: Hardcode for Emergency Fix

If you need `BioDockify.in` working IMMEDIATELY while troubleshooting Vercel:

**File:** `web/src/supabaseClient.js`
```javascript
import { createClient } from '@supabase/supabase-js'

// TEMPORARY: Replace with your actual values
const supabaseUrl = 'https://YOUR-PROJECT-ID.supabase.co'
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.YOUR-ACTUAL-KEY'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
```

Then:
```bash
git add web/src/supabaseClient.js
git commit -m "temp: hardcode supabase credentials for production"
git push
```

**⚠️ IMPORTANT:** This is NOT recommended long-term (security risk). Only use as emergency fix, then revert once environment variables work properly.
