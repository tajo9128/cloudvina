-- Add user verification and organization fields
-- Run this in Supabase SQL Editor

-- Add columns to auth.users metadata (handled by Supabase Auth)
-- We'll use user_metadata for phone, designation, organization

-- Create user_profiles table for additional info
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    phone VARCHAR(15),
    phone_verified BOOLEAN DEFAULT FALSE,
    designation VARCHAR(100),
    organization VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add RLS policies
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid() = id);

-- Create function to track daily job usage
CREATE TABLE IF NOT EXISTS daily_job_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    job_date DATE DEFAULT CURRENT_DATE,
    job_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, job_date)
);

-- RLS for daily_job_usage
ALTER TABLE daily_job_usage ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own usage" ON daily_job_usage
    FOR SELECT USING (auth.uid() = user_id);

-- Function to check if user can submit job
CREATE OR REPLACE FUNCTION can_submit_job(p_user_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_user_plan TEXT;
    v_email_verified BOOLEAN;
    v_phone_verified BOOLEAN;
    v_today_count INTEGER;
    v_free_limit INTEGER := 3; -- 3 jobs per day for free users
BEGIN
    -- Check email verification
    SELECT email_confirmed_at IS NOT NULL INTO v_email_verified
    FROM auth.users
    WHERE id = p_user_id;
    
    IF NOT v_email_verified THEN
        RETURN FALSE;
    END IF;
    
    -- Check phone verification
    SELECT phone_verified INTO v_phone_verified
    FROM user_profiles
    WHERE id = p_user_id;
    
    IF v_phone_verified IS NULL OR NOT v_phone_verified THEN
        RETURN FALSE;
    END IF;
    
    -- Get user's current plan
    SELECT plan INTO v_user_plan
    FROM user_credits
    WHERE user_id = p_user_id;
    
    -- If not free plan, allow (paid users have no daily limit)
    IF v_user_plan IS NOT NULL AND v_user_plan != 'free' THEN
        RETURN TRUE;
    END IF;
    
    -- Check today's usage for free users
    SELECT COALESCE(job_count, 0) INTO v_today_count
    FROM daily_job_usage
    WHERE user_id = p_user_id AND job_date = CURRENT_DATE;
    
    -- Return true if under limit
    RETURN v_today_count < v_free_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to increment daily usage
CREATE OR REPLACE FUNCTION increment_daily_usage(p_user_id UUID)
RETURNS VOID AS $$
BEGIN
    INSERT INTO daily_job_usage (user_id, job_date, job_count)
    VALUES (p_user_id, CURRENT_DATE, 1)
    ON CONFLICT (user_id, job_date)
    DO UPDATE SET job_count = daily_job_usage.job_count + 1;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_daily_usage_user_date ON daily_job_usage(user_id, job_date);
CREATE INDEX IF NOT EXISTS idx_user_profiles_phone ON user_profiles(phone);

-- Add plan column to user_credits if it doesn't exist
ALTER TABLE user_credits ADD COLUMN IF NOT EXISTS plan VARCHAR(50) DEFAULT 'free';
