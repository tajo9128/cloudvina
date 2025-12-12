-- Fix: Final Resolution for Profiles/User_Profiles Conflict (NO PHONE VERSION)
-- Context: User removed phone verification. Resolving schema collision.

-- 1. Ensure 'profiles' table has required columns (Excluding phone)
ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS designation VARCHAR(100),
ADD COLUMN IF NOT EXISTS organization VARCHAR(200),
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS avatar_url TEXT,
ADD COLUMN IF NOT EXISTS username TEXT,
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- 2. Migrate Data from 'user_profiles' (Safely Merge)
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'user_profiles') THEN
        
        -- A. Update existing profiles with data from user_profiles (if they match on ID)
        UPDATE public.profiles p
        SET 
            designation = up.designation,
            organization = up.organization
        FROM public.user_profiles up
        WHERE p.id = up.id;

        -- B. Insert missing profiles from user_profiles
        INSERT INTO public.profiles (id, designation, organization, created_at, updated_at)
        SELECT id, designation, organization, created_at, updated_at
        FROM public.user_profiles up
        WHERE NOT EXISTS (SELECT 1 FROM public.profiles p WHERE p.id = up.id);

        -- C. Drop the duplicate table safely
        DROP TABLE public.user_profiles CASCADE;
        
    END IF;
END $$;

-- 3. Ensure RLS is active and correct
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Re-apply policies (Idempotent)
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

DROP POLICY IF EXISTS "Admins can view all profiles" ON public.profiles;
CREATE POLICY "Admins can view all profiles" ON public.profiles
    FOR SELECT USING (is_admin = TRUE);

-- 4. Verify Trigger Function (No Phone)
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, designation, organization, is_admin)
  VALUES (
    new.id,
    new.raw_user_meta_data->>'designation',
    new.raw_user_meta_data->>'organization',
    FALSE
  );
  -- user_credits logic (unchanged)
  INSERT INTO public.user_credits (
    user_id, plan, credits, bonus_credits, bonus_expiry, 
    monthly_credits, last_monthly_reset, paid_credits, account_created_at
  )
  VALUES (
    new.id, 'free', 130, 100, NOW() + INTERVAL '30 days', 
    30, CURRENT_DATE, 0, NOW()
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
