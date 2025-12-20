-- Benchmark/Accuracy Analysis Table
-- Stores validation reports (Planned vs Actual)

CREATE TABLE IF NOT EXISTS public.benchmark_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    batch_id TEXT NOT NULL, -- Logical link to jobs.batch_id
    name TEXT NOT NULL, -- Report Name (e.g. "PDBbind Validation 1")
    dataset_filename TEXT, -- Name of the uploaded CSV
    metrics JSONB DEFAULT '{}'::jsonb, -- { "r2": 0.8, "rmse": 1.2, "n": 50 }
    plot_data JSONB DEFAULT '[]'::jsonb, -- Cached scatter plot points for swift rendering
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- RLS
ALTER TABLE public.benchmark_analyses ENABLE ROW LEVEL SECURITY;

-- Users see their own analyses
CREATE POLICY "Users view own benchmarks" ON public.benchmark_analyses
    FOR SELECT USING (auth.uid() = user_id);

-- Users create their own analyses
CREATE POLICY "Users insert own benchmarks" ON public.benchmark_analyses
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users delete their own
CREATE POLICY "Users delete own benchmarks" ON public.benchmark_analyses
    FOR DELETE USING (auth.uid() = user_id);
