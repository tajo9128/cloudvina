-- CloudVina Pricing Plans
-- Run this in Supabase SQL Editor to add recommended pricing

-- Insert pricing tiers
INSERT INTO pricing_plans (name, price_monthly, price_yearly, credits_per_month, description, features, is_active) VALUES

-- Free Tier
('Free Trial', 0, 0, 10, 
 'Perfect for trying molecular docking',
 ARRAY[
   '10 free docking jobs',
   'Valid for 30 days',
   'Basic support',
   'Community access',
   'Download results'
 ],
 true),

-- Student Tier
('Student Plan', 99, 999, 100,
 'Ideal for students and researchers',
 ARRAY[
   '100 docking jobs/month',
   '~3 jobs per day',
   'Priority email support',
   'Advanced parameters',
   'Export to PDF/CSV',
   'Activity logs',
   '50% discount on annual plan'
 ],
 true),

-- Researcher Tier
('Researcher Plan', 499, 4999, 500,
 'For active researchers and labs',
 ARRAY[
   '500 docking jobs/month',
   'Priority processing',
   'Dedicated support',
   'API access',
   'Batch uploads',
   'Analytics dashboard',
   'Custom workflows',
   '2 months free on annual plan'
 ],
 true),

-- Institution Tier
('Institution Plan', 3999, 39999, 5000,
 'For universities and research institutions',
 ARRAY[
   '5,000 shared credits/month',
   'Multi-user access',
   'Priority processing',
   'Dedicated support channel',
   'Custom integrations',
   'Training sessions',
   'Admin dashboard',
   'SLA guarantee',
   '3 months free on annual plan'
 ],
 true)

ON CONFLICT (name) DO UPDATE SET
  price_monthly = EXCLUDED.price_monthly,
  price_yearly = EXCLUDED.price_yearly,
  credits_per_month = EXCLUDED.credits_per_month,
  description = EXCLUDED.description,
  features = EXCLUDED.features,
  is_active = EXCLUDED.is_active;

-- Verify
SELECT name, price_monthly, price_yearly, credits_per_month, description 
FROM pricing_plans 
ORDER BY price_monthly;
