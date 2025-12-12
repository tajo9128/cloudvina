# Supabase Setup Guide for learn.biodockify.com

## Step 1: Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Sign in / Create account
3. Click **"New Project"**
4. Configure:
   - **Name**: biodockify-learning
   - **Database Password**: (generate strong password and save it!)
   - **Region**: Choose closest to your users
   - **Plan**: Free tier (upgrade later if needed)
5. Wait 2-3 minutes for provisioning

## Step 2: Run Database Schema

1. In Supabase dashboard, go to **SQL Editor**
2. Click **"New query"**
3. Copy entire content from `supabase_schema.sql`
4. Paste and click **"Run"**
5. Verify tables created in **Table Editor**

## Step 3: Get API Credentials

1. Go to **Settings → API**
2. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon/public key**: `eyJhbGc...` (long string)
   - **service_role key**: (keep secret, for admin operations)

## Step 4: Configure Environment Variables

Create `.env.local` in `web_learn/`:

```env
VITE_SUPABASE_URL=https://xxxxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**IMPORTANT**: Add `.env.local` to `.gitignore`!

## Step 5: Enable Authentication

1. Go to **Authentication → Providers**
2. Enable:
   - **Email** (enabled by default)
   - **Google** (optional - configure OAuth)
   - **GitHub** (optional - configure OAuth)

3. Configure **Email Templates**:
   - Go to **Authentication → Email Templates**
   - Customize confirmation email

## Step 6: Storage Setup (for images/videos)

1. Go to **Storage**
2. Create buckets:
   - `course-thumbnails` (public)
   - `lesson-videos` (public or authenticated)
   - `user-avatars` (public)

3. Set policies:
```sql
-- Allow public read for thumbnails
CREATE POLICY "Public Access"
ON storage.objects FOR SELECT
USING ( bucket_id = 'course-thumbnails' );

-- Allow authenticated users to upload avatars
CREATE POLICY "Users can upload avatars"
ON storage.objects FOR INSERT
WITH CHECK (
  bucket_id = 'user-avatars' 
  AND auth.uid()::text = (storage.foldername(name))[1]
);
```

## Step 7: Seed Sample Data (Optional)

Run this to create sample course:

```sql
-- Create instructor profile
INSERT INTO user_profiles (id, username, display_name, role)
VALUES (auth.uid(), 'instructor1', 'Dr. BioDock', 'instructor');

-- Create sample course
INSERT INTO courses (title, slug, description, instructor_id, difficulty, is_published)
VALUES (
  'Introduction to Molecular Docking',
  'intro-molecular-docking',
  'Learn the fundamentals of molecular docking with AutoDock Vina',
  auth.uid(),
  'beginner',
  true
);

-- Get course ID
DO $$
DECLARE
  v_course_id UUID;
  v_module_id UUID;
BEGIN
  SELECT id INTO v_course_id FROM courses WHERE slug = 'intro-molecular-docking';
  
  -- Create module
  INSERT INTO course_modules (course_id, title, order_index)
  VALUES (v_course_id, 'Getting Started', 1)
  RETURNING id INTO v_module_id;
  
  -- Create lessons
  INSERT INTO lessons (module_id, title, slug, content, order_index, is_free)
  VALUES 
    (v_module_id, 'Welcome', 'welcome', '# Welcome\n\nWelcome to the course!', 1, true),
    (v_module_id, 'What is Docking?', 'what-is-docking', '# What is Docking?\n\nMolecular docking explained...', 2, true);
END $$;
```

## Step 8: Test Connection

Test in browser console:
```javascript
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  'YOUR_SUPABASE_URL',
  'YOUR_ANON_KEY'
)

// Test query
const { data, error } = await supabase
  .from('courses')
  .select('*')
  
console.log(data, error)
```

## Next Steps

- ✅ Database schema created
- ✅ Authentication configured
- ✅ Storage buckets set up
- ⏭️ Install Supabase client in React app
- ⏭️ Build course components

## Troubleshooting

**Tables not appearing?**
- Check SQL Editor for errors
- Verify you ran entire schema script

**RLS errors?**
- Make sure RLS policies are created
- Test with authenticated user

**Can't connect?**
- Verify API keys in `.env.local`
- Check Supabase project status

---

Your Supabase backend is now ready for the learning platform! 🎉
