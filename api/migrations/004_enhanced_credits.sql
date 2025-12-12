-- Enhanced Credit System with Time-Based Daily Limits
-- Run this in Supabase SQL Editor

-- Step 1: Add columns to user_credits
ALTER TABLE user_credits 
ADD COLUMN IF NOT EXISTS bonus_credits INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS bonus_expiry TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS monthly_credits INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_monthly_reset DATE DEFAULT CURRENT_DATE,
ADD COLUMN IF NOT EXISTS paid_credits INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS account_created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Step 2: Function to get daily limit based on account age
CREATE OR REPLACE FUNCTION get_daily_limit(p_user_id UUID)
RETURNS INTEGER AS $$
DECLARE
    v_plan TEXT;
    v_created_at TIMESTAMP;
    v_age_days INTEGER;
BEGIN
    SELECT plan, account_created_at INTO v_plan, v_created_at
    FROM user_credits
    WHERE user_id = p_user_id;
    
    -- Paid users: no daily limit
    IF v_plan IS NOT NULL AND v_plan != 'free' THEN
        RETURN 99999; -- Effectively unlimited
    END IF;
    
    -- Free users: check account age
    v_age_days := EXTRACT(DAY FROM NOW() - v_created_at)::INTEGER;
    
    -- First 30 days: 3 jobs/day
    IF v_age_days < 30 THEN
        RETURN 3;
    ELSE
        -- After 30 days: 1 job/day
        RETURN 1;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Step 3: Function to get available credits
CREATE OR REPLACE FUNCTION get_available_credits(p_user_id UUID)
RETURNS TABLE(
    total_credits INTEGER,
    bonus_credits INTEGER,
    monthly_credits INTEGER,
    paid_credits INTEGER,
    plan TEXT
) AS $$
DECLARE
    v_bonus INTEGER := 0;
    v_monthly INTEGER := 0;
    v_paid INTEGER := 0;
    v_plan TEXT;
    v_bonus_expiry TIMESTAMP;
    v_last_reset DATE;
BEGIN
    -- Get user credits info
    SELECT 
        COALESCE(uc.bonus_credits, 0),
        COALESCE(uc.bonus_expiry, NOW()),
        COALESCE(uc.monthly_credits, 0),
        COALESCE(uc.last_monthly_reset, CURRENT_DATE),
        COALESCE(uc.paid_credits, 0),
        COALESCE(uc.plan, 'free')
    INTO v_bonus, v_bonus_expiry, v_monthly, v_last_reset, v_paid, v_plan
    FROM user_credits uc
    WHERE uc.user_id = p_user_id;
    
    -- Check if bonus expired
    IF v_bonus_expiry < NOW() THEN
        v_bonus := 0;
    END IF;
    
    -- Check if monthly needs reset (for free users)
    IF v_last_reset < CURRENT_DATE AND v_plan = 'free' THEN
        v_monthly := 30; -- Reset to 30 monthly credits
    END IF;
    
    -- Return credits breakdown
    RETURN QUERY SELECT 
        (v_bonus + v_monthly + v_paid)::INTEGER as total,
        v_bonus::INTEGER,
        v_monthly::INTEGER,
        v_paid::INTEGER,
        v_plan;
END;
$$ LANGUAGE plpgsql;

-- Step 4: Function to deduct credits (uses bonus first, then monthly, then paid)
CREATE OR REPLACE FUNCTION deduct_credit(p_user_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_bonus INTEGER;
    v_monthly INTEGER;
    v_paid INTEGER;
    v_plan TEXT;
    v_bonus_expiry TIMESTAMP;
BEGIN
    -- Get current credits
    SELECT 
        COALESCE(bonus_credits, 0),
        COALESCE(monthly_credits, 0),
        COALESCE(paid_credits, 0),
        COALESCE(plan, 'free'),
        bonus_expiry
    INTO v_bonus, v_monthly, v_paid, v_plan, v_bonus_expiry
    FROM user_credits
    WHERE user_id = p_user_id;
    
    -- Check if bonus expired
    IF v_bonus_expiry IS NOT NULL AND v_bonus_expiry < NOW() THEN
        v_bonus := 0;
        UPDATE user_credits SET bonus_credits = 0 WHERE user_id = p_user_id;
    END IF;
    
    -- Deduct in order: bonus -> monthly -> paid
    IF v_bonus > 0 THEN
        UPDATE user_credits 
        SET bonus_credits = bonus_credits - 1, credits = credits - 1
        WHERE user_id = p_user_id;
        RETURN TRUE;
    ELSIF v_monthly > 0 THEN
        UPDATE user_credits 
        SET monthly_credits = monthly_credits - 1, credits = credits - 1
        WHERE user_id = p_user_id;
        RETURN TRUE;
    ELSIF v_paid > 0 THEN
        UPDATE user_credits 
        SET paid_credits = paid_credits - 1, credits = credits - 1
        WHERE user_id = p_user_id;
        
        -- Check if paid credits exhausted, downgrade to free
        IF v_paid - 1 = 0 AND v_plan != 'free' THEN
            UPDATE user_credits 
            SET plan = 'free', 
                monthly_credits = 30,
                last_monthly_reset = CURRENT_DATE,
                account_created_at = NOW() -- Reset for 30-day 3/day limit
            WHERE user_id = p_user_id;
        END IF;
        
        RETURN TRUE;
    ELSE
        RETURN FALSE; -- No credits available
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Step 5: Function to reset monthly credits (run daily via cron)
CREATE OR REPLACE FUNCTION reset_monthly_credits()
RETURNS INTEGER AS $$
DECLARE
    rows_updated INTEGER;
BEGIN
    UPDATE user_credits
    SET monthly_credits = 30,
        last_monthly_reset = CURRENT_DATE
    WHERE plan = 'free' 
    AND last_monthly_reset < CURRENT_DATE;
    
    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    RETURN rows_updated;
END;
$$ LANGUAGE plpgsql;

-- Step 6: Update existing free users
UPDATE user_credits
SET 
    bonus_credits = CASE WHEN credits > 0 THEN LEAST(credits, 100) ELSE 0 END,
    bonus_expiry = NOW() + INTERVAL '30 days',
    monthly_credits = 30,
    last_monthly_reset = CURRENT_DATE,
    paid_credits = 0,
    account_created_at = COALESCE(created_at, NOW())
WHERE plan = 'free' OR plan IS NULL;

-- Step 7: Update existing paid users
UPDATE user_credits
SET 
    paid_credits = credits,
    bonus_credits = 0,
    monthly_credits = 0,
    account_created_at = COALESCE(created_at, NOW())
WHERE plan != 'free' AND plan IS NOT NULL;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_daily_usage_user_date ON daily_job_usage(user_id, job_date);
CREATE INDEX IF NOT EXISTS idx_user_profiles_phone ON user_profiles(phone);
CREATE INDEX IF NOT EXISTS idx_user_credits_plan ON user_credits(plan);

-- Verify
SELECT 
    user_id,
    plan,
    bonus_credits,
    bonus_expiry::DATE as bonus_expires,
    monthly_credits,
    paid_credits,
    EXTRACT(DAY FROM NOW() - account_created_at)::INTEGER as account_age_days,
    get_daily_limit(user_id) as daily_limit
FROM user_credits
LIMIT 5;
