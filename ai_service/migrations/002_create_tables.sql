-- Phase 2: AI Service Tables (AI Suffix Applied)
-- Run this in Supabase SQL Editor

-- 1. Projects table (AI)
CREATE TABLE IF NOT EXISTS projects_ai (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID, -- Linked to auth.users if possible, or stored as raw UUID
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- 2. Compounds table (AI)
CREATE TABLE IF NOT EXISTS compounds_ai (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects_ai(id) ON DELETE CASCADE,
    smiles VARCHAR(1000) NOT NULL,
    chem_name VARCHAR(255),
    source VARCHAR(100) DEFAULT 'upload',
    properties JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- 3. QSAR Models table (AI)
CREATE TABLE IF NOT EXISTS qsar_models_ai (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects_ai(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'regression',
    target_column VARCHAR(255),
    metrics JSONB DEFAULT '{}'::jsonb,
    model_path VARCHAR(500) NOT NULL,
    status VARCHAR(50) DEFAULT 'ready',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- 4. Predictions table (AI)
CREATE TABLE IF NOT EXISTS predictions_ai (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES qsar_models_ai(id) ON DELETE CASCADE,
    smiles VARCHAR(1000) NOT NULL,
    result FLOAT NOT NULL,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_projects_ai_user_id ON projects_ai(user_id);
CREATE INDEX IF NOT EXISTS idx_compounds_ai_project_id ON compounds_ai(project_id);
