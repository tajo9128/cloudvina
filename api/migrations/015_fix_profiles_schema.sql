-- Fix: Align Schema for Legacy Web App (profiles table)
-- Resolves 406 Error on "select=is_admin" checks

-- 1. Standardize on 'profiles' table name
-- If user_profiles exists from our recent setup, rename it.
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'user_profiles') THEN
        ALTER TABLE public.user_profiles RENAME TO profiles;
    END IF;
END $$;

-- 2. Ensure 'profiles' table exists (if user_profiles didn't exist)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    phone VARCHAR(15),
    phone_verified BOOLEAN DEFAULT FALSE,
    designation VARCHAR(100),
    organization VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Add columns expected by Legacy App / Admin Queries
ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS avatar_url TEXT,
ADD COLUMN IF NOT EXISTS username TEXT;

-- 4. Fix RLS Policies for 'profiles'
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Re-create policies using the new table name
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

DROP POLICY IF EXISTS "Admins can view all profiles" ON public.profiles;
CREATE POLICY "Admins can view all profiles" ON public.profiles
    FOR SELECT USING (is_admin = TRUE);

-- 5. Update Signup Trigger to use 'profiles'
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, phone, designation, organization, phone_verified, is_admin)
  VALUES (
    new.id,
    new.raw_user_meta_data->>'phone',
    new.raw_user_meta_data->>'designation',
    new.raw_user_meta_data->>'organization',
    FALSE,
    FALSE
  );

  -- Handle user_credits (exists in both setups)
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
