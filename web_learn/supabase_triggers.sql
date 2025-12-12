-- Trigger to automatically create user_profiles record on new user sign up
-- Run this in Supabase SQL Editor

-- 1. Create Function
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.lms_profiles (id, username, display_name, avatar_url, role)
  VALUES (
    NEW.id,
    LOWER(SPLIT_PART(NEW.email, '@', 1)), -- Default username from email
    NEW.raw_user_meta_data->>'full_name', -- Display Name from metadata
    'https://api.dicebear.com/7.x/avataaars/svg?seed=' || NEW.id, -- Default avatar
    'student' -- Default role
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 2. Create Trigger
CREATE OR REPLACE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
