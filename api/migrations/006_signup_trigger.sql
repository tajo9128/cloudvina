-- Automate User Setup with Triggers
-- Run this in Supabase SQL Editor

-- 1. Create the handler function
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

-- 2. Create the trigger
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();
