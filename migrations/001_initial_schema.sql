-- AI Service Migration (Comprehensive)
-- Run this in Supabase SQL Editor

-- 1. AI Projects Table (Hierarchy Root)
CREATE TABLE IF NOT EXISTS public.ai_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- 2. AI Compounds Table (Dataset)
CREATE TABLE IF NOT EXISTS public.ai_compounds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES public.ai_projects(id) ON DELETE CASCADE,
    smiles TEXT NOT NULL,
    chem_name VARCHAR(255),
    source VARCHAR(100), -- 'upload', 'generated'
    properties JSONB, -- { "logP": 2.5, "mw": 350.1 }
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- 3. AI QSAR Models Table
CREATE TABLE IF NOT EXISTS public.ai_qsar_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES public.ai_projects(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE, -- Denormalized for easy RLS
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'regression',
    target_column VARCHAR(255),
    algorithm VARCHAR(50), 
    metrics JSONB, -- { "r2": 0.85, "rmse": 0.12 }
    model_path VARCHAR(255), -- S3 or HF Dataset path
    status VARCHAR(50) DEFAULT 'training', 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- 4. AI Training Jobs
CREATE TABLE IF NOT EXISTS public.ai_training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES public.ai_projects(id) ON DELETE CASCADE,
    model_id UUID REFERENCES public.ai_qsar_models(id),
    status VARCHAR(50) DEFAULT 'queued', -- 'queued', 'running', 'completed', 'failed'
    progress INT DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- 5. AI Predictions
CREATE TABLE IF NOT EXISTS public.ai_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES public.ai_qsar_models(id) ON DELETE CASCADE,
    compound_id UUID REFERENCES public.ai_compounds(id), 
    input_smiles TEXT, -- If not linked to existing compound
    prediction_value FLOAT,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Enable RLS on all tables
ALTER TABLE public.ai_projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_compounds ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_qsar_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_training_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_predictions ENABLE ROW LEVEL SECURITY;

-- RLS Policies (Simplified for Owner Access)
-- Projects
CREATE POLICY "Owner access projects" ON public.ai_projects 
USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Compounds (via Project)
CREATE POLICY "Owner access compounds" ON public.ai_compounds 
USING (EXISTS (SELECT 1 FROM public.ai_projects WHERE id = project_id AND user_id = auth.uid())) 
WITH CHECK (EXISTS (SELECT 1 FROM public.ai_projects WHERE id = project_id AND user_id = auth.uid()));

-- QSAR Models
CREATE POLICY "Owner access models" ON public.ai_qsar_models 
USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Training Jobs
CREATE POLICY "Owner access jobs" ON public.ai_training_jobs 
USING (EXISTS (SELECT 1 FROM public.ai_projects WHERE id = project_id AND user_id = auth.uid()));

-- Predictions
CREATE POLICY "Owner access predictions" ON public.ai_predictions 
USING (EXISTS (SELECT 1 FROM public.ai_qsar_models WHERE id = model_id AND user_id = auth.uid()));
