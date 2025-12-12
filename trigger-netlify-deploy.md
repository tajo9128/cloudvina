# Trigger Netlify Deploy

## Quick Fix for BioDockify.in Not Updating

### Method 1: Via Netlify Dashboard (Fastest)
1. Go to https://app.netlify.com/
2. Select **BioDockify** site
3. Click **Deploys** tab
4. Click **"Trigger deploy"** → **"Clear cache and deploy site"**

### Method 2: Using Build Hook (One-time setup)

1. In Netlify Dashboard → **Site Settings** → **Build & Deploy** → **Build Hooks**
2. Create a new hook called "Manual Deploy"
3. Copy the webhook URL
4. Save it as an environment variable or use this PowerShell command:

```powershell
# Replace <YOUR_HOOK_URL> with the actual URL from Netlify
Invoke-WebRequest -Uri "<YOUR_HOOK_URL>" -Method POST
```

### Method 3: Check Auto-Deploy Settings
Make sure Netlify is watching the correct branch:
1. **Site Settings** → **Build & Deploy** → **Continuous Deployment**
2. Verify **Branch to deploy** is set to `main`
3. Ensure **Auto publishing** is enabled

---

## Why This Happens
- **Vercel** (BioDockify-3n3v.vercel.app) auto-deploys from every commit
- **Netlify** (BioDockify.in) might have:
  - Build failures (check deploy logs)
  - Different branch configured
  - Caching issues

## Prevention
Consider using **only one hosting service** to avoid this confusion:
- Keep Vercel for preview URLs
- Move production (`BioDockify.in`) to Vercel OR
- Move everything to Netlify
