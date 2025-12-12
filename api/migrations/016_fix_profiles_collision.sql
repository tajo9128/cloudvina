-- Fix: Resolve "profiles already exists" error
-- Handles collision between 'user_profiles' and 'profiles' tables

DO $$
BEGIN
    -- 1. If 'user_profiles' exists AND 'profiles' exists:
    --    Merge data from user_profiles to profiles, then drop user_profiles.
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'user_profiles')
    AND EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'profiles') THEN
        
        -- Copy data from user_profiles to profiles where id doesn't exist
        INSERT INTO public.profiles (id, phone, phone_verified, designation, organization, created_at, updated_at)
        SELECT id, phone, phone_verified, designation, organization, created_at, updated_at
        FROM public.user_profiles
        ON CONFLICT (id) DO NOTHING;

        -- Drop the old table
        DROP TABLE public.user_profiles CASCADE;
    
    -- 2. If ONLY 'user_profiles' exists (renaming case)
    ELSIF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'user_profiles') THEN
        ALTER TABLE public.user_profiles RENAME TO profiles;
    END IF;

    -- 3. If only 'profiles' exists (do nothing, falling through to column checks)
END $$;

-- 4. Ensure 'profiles' table structure is correct
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    phone VARCHAR(15),
    phone_verified BOOLEAN DEFAULT FALSE,
    designation VARCHAR(100),
    organization VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Add columns expected by Legacy App / Admin Queries (Safe if they already exist)
ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS avatar_url TEXT,
ADD COLUMN IF NOT EXISTS username TEXT,
ADD COLUMN IF NOT EXISTS phone VARCHAR(15),
ADD COLUMN IF NOT EXISTS phone_verified BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS designation VARCHAR(100),
ADD COLUMN IF NOT EXISTS organization VARCHAR(200);

-- 6. Reset RLS Policies for 'profiles' (Ensure they are correct)
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

DROP POLICY IF EXISTS "Admins can view all profiles" ON public.profiles;
CREATE POLICY "Admins can view all profiles" ON public.profiles
    FOR SELECT USING (is_admin = TRUE);

-- 7. Update Signup Trigger to use 'profiles'
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
