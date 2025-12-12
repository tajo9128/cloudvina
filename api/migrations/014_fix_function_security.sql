-- Fix: Function Search Path Mutable (Security Hardening)
-- Detected by Supabase Linter

-- 1. Secure `handle_new_user`
-- Prevents malicious search_path manipulation during user creation
ALTER FUNCTION public.handle_new_user()
SET search_path = public;

-- 2. Secure `get_daily_limit`
-- Ensures consistent schema usage for logic
ALTER FUNCTION public.get_daily_limit(user_uuid uuid)
SET search_path = public;

-- 3. Secure `reset_monthly_credits`
-- Critical for cron jobs to run safely
ALTER FUNCTION public.reset_monthly_credits()
SET search_path = public;

-- NOTE: "Leaked Password Protection" can only be enabled via Supabase Dashboard:
-- Authentication -> Configuration -> Security -> [x] Enable Leaked Password Protection
