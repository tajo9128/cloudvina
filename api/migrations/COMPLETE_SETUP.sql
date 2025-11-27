-- ============================================================================
-- COMPLETE DATABASE SETUP FOR CLOUDVINA
-- Run this entire script in Supabase SQL Editor if tables don't exist
-- ============================================================================

-- 1. CREATE USER_PROFILES TABLE
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    phone VARCHAR(15),
    phone_verified BOOLEAN DEFAULT FALSE,
    designation VARCHAR(100),
    organization VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. CREATE DAILY_JOB_USAGE TABLE
CREATE TABLE IF NOT EXISTS daily_job_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    job_date DATE DEFAULT CURRENT_DATE,
    job_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, job_date)
);

-- 3. ADD RLS POLICIES FOR USER_PROFILES
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own profile" ON user_profiles;
CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON user_profiles;
CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid() = id);

-- 4. ADD RLS POLICIES FOR DAILY_JOB_USAGE
ALTER TABLE daily_job_usage ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own usage" ON daily_job_usage;
CREATE POLICY "Users can view own usage" ON daily_job_usage
    FOR SELECT USING (auth.uid() = user_id);

-- 5. GRANT PERMISSIONS (Important for API access)
GRANT ALL ON public.user_profiles TO postgres;
GRANT ALL ON public.user_profiles TO service_role;
GRANT ALL ON public.user_profiles TO authenticated;

GRANT ALL ON public.daily_job_usage TO postgres;
GRANT ALL ON public.daily_job_usage TO service_role;
GRANT ALL ON public.daily_job_usage TO authenticated;

-- 6. CREATE INDEXES FOR PERFORMANCE
CREATE INDEX IF NOT EXISTS idx_daily_usage_user_date ON daily_job_usage(user_id, job_date);
CREATE INDEX IF NOT EXISTS idx_user_profiles_phone ON user_profiles(phone);

-- 7. CREATE SIGNUP TRIGGER (Auto-create user profile on signup)
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  -- Insert into user_profiles
  INSERT INTO public.user_profiles (id, phone, designation, organization, phone_verified)
  VALUES (
    new.id,
    new.raw_user_meta_data->>'phone',
    new.raw_user_meta_data->>'designation',
    new.raw_user_meta_data->>'organization',
    FALSE
  );

  -- Insert into user_credits
  INSERT INTO public.user_credits (
    user_id, 
    plan, 
    credits, 
    bonus_credits, 
    bonus_expiry, 
    monthly_credits, 
    last_monthly_reset, 
    paid_credits, 
    account_created_at
  )
  VALUES (
    new.id,
    'free',
    130, -- Total (100 bonus + 30 monthly)
    100, -- Bonus
    NOW() + INTERVAL '30 days',
    30, -- Monthly
    CURRENT_DATE,
    0,
    NOW()
  );

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 8. CREATE TRIGGER
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();

-- 9. RELOAD SCHEMA CACHE
NOTIFY pgrst, 'reload config';

-- ============================================================================
-- VERIFICATION QUERIES (Run these to check if everything worked)
-- ============================================================================

-- Check if tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('user_profiles', 'daily_job_usage');

-- Check if policies exist
SELECT tablename, policyname 
FROM pg_policies 
WHERE tablename IN ('user_profiles', 'daily_job_usage');

-- Check if trigger exists
SELECT trigger_name, event_object_table 
FROM information_schema.triggers 
WHERE trigger_name = 'on_auth_user_created';
