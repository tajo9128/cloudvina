-- BioDockify Learning Platform - Database Schema
-- Course System Tables (Namespaced with lms_)

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- USER PROFILES (lms_profiles)
-- ============================================
CREATE TABLE lms_profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  username VARCHAR(50) UNIQUE NOT NULL,
  display_name VARCHAR(100),
  avatar_url TEXT,
  bio TEXT,
  role VARCHAR(20) DEFAULT 'student', -- student, instructor, admin
  reputation INT DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- COURSE SYSTEM
-- ============================================

-- Courses table
CREATE TABLE lms_courses (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  title VARCHAR(200) NOT NULL,
  slug VARCHAR(200) UNIQUE NOT NULL,
  description TEXT,
  thumbnail_url TEXT,
  instructor_id UUID REFERENCES lms_profiles(id),
  difficulty VARCHAR(20) CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
  duration_hours INT,
  price DECIMAL(10,2) DEFAULT 0,
  is_published BOOLEAN DEFAULT false,
  is_featured BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Course modules
CREATE TABLE lms_modules (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  course_id UUID REFERENCES lms_courses(id) ON DELETE CASCADE,
  title VARCHAR(200) NOT NULL,
  description TEXT,
  order_index INT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(course_id, order_index)
);

-- Lessons
CREATE TABLE lms_lessons (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  module_id UUID REFERENCES lms_modules(id) ON DELETE CASCADE,
  title VARCHAR(200) NOT NULL,
  slug VARCHAR(200) NOT NULL,
  content TEXT, -- Markdown/HTML content
  video_url TEXT,
  duration_minutes INT,
  order_index INT NOT NULL,
  is_free BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(module_id, order_index)
);

-- Quizzes
CREATE TABLE lms_quizzes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  lesson_id UUID REFERENCES lms_lessons(id) ON DELETE CASCADE,
  title VARCHAR(200) NOT NULL,
  description TEXT,
  passing_score INT DEFAULT 70,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Quiz questions
CREATE TABLE lms_quiz_questions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  quiz_id UUID REFERENCES lms_quizzes(id) ON DELETE CASCADE,
  question TEXT NOT NULL,
  question_type VARCHAR(20) CHECK (question_type IN ('multiple_choice', 'true_false', 'short_answer')),
  options JSONB,
  explanation TEXT,
  order_index INT NOT NULL,
  points INT DEFAULT 1,
  UNIQUE(quiz_id, order_index)
);

-- ============================================
-- USER PROGRESS & ENROLLMENT
-- ============================================

-- Course enrollments
CREATE TABLE lms_enrollments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES lms_profiles(id) ON DELETE CASCADE,
  course_id UUID REFERENCES lms_courses(id) ON DELETE CASCADE,
  enrolled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  completed_at TIMESTAMP WITH TIME ZONE,
  progress_percent INT DEFAULT 0,
  last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, course_id)
);

-- Lesson completion tracking
CREATE TABLE lms_completions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES lms_profiles(id) ON DELETE CASCADE,
  lesson_id UUID REFERENCES lms_lessons(id) ON DELETE CASCADE,
  completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, lesson_id)
);

-- Quiz attempts
CREATE TABLE lms_quiz_attempts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES lms_profiles(id) ON DELETE CASCADE,
  quiz_id UUID REFERENCES lms_quizzes(id) ON DELETE CASCADE,
  score INT NOT NULL,
  max_score INT NOT NULL,
  passed BOOLEAN NOT NULL,
  answers JSONB,
  completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Certificates
CREATE TABLE lms_certificates (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES lms_profiles(id) ON DELETE CASCADE,
  course_id UUID REFERENCES lms_courses(id) ON DELETE CASCADE,
  certificate_number VARCHAR(50) UNIQUE NOT NULL,
  certificate_url TEXT,
  issued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, course_id)
);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX idx_lms_courses_slug ON lms_courses(slug);
CREATE INDEX idx_lms_courses_published ON lms_courses(is_published);
CREATE INDEX idx_lms_lessons_module ON lms_lessons(module_id);
CREATE INDEX idx_lms_enrollments_user ON lms_enrollments(user_id);
CREATE INDEX idx_lms_completions_user ON lms_completions(user_id);

-- ============================================
-- RLS Policies (Updated for lms_ tables)
-- ============================================

ALTER TABLE lms_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE lms_courses ENABLE ROW LEVEL SECURITY;
ALTER TABLE lms_modules ENABLE ROW LEVEL SECURITY;
ALTER TABLE lms_lessons ENABLE ROW LEVEL SECURITY;
ALTER TABLE lms_enrollments ENABLE ROW LEVEL SECURITY;
ALTER TABLE lms_completions ENABLE ROW LEVEL SECURITY;
ALTER TABLE lms_quiz_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE lms_certificates ENABLE ROW LEVEL SECURITY;

-- Profiles
CREATE POLICY "Public profiles" ON lms_profiles FOR SELECT USING (true);
CREATE POLICY "Update own profile" ON lms_profiles FOR UPDATE USING (auth.uid() = id);

-- Courses
CREATE POLICY "View published courses" ON lms_courses FOR SELECT USING (is_published = true OR auth.uid() = instructor_id);
CREATE POLICY "Instructor create courses" ON lms_courses FOR INSERT WITH CHECK (auth.uid() = instructor_id);
CREATE POLICY "Instructor update courses" ON lms_courses FOR UPDATE USING (auth.uid() = instructor_id);

-- Enrollments
CREATE POLICY "View own enrollments" ON lms_enrollments FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Create enrollment" ON lms_enrollments FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Completions
CREATE POLICY "View own completions" ON lms_completions FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Mark complete" ON lms_completions FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Functions (Updated)
CREATE OR REPLACE FUNCTION update_updated_at_column() RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$ LANGUAGE plpgsql;

CREATE TRIGGER update_lms_courses_mod BEFORE UPDATE ON lms_courses FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_lms_lessons_mod BEFORE UPDATE ON lms_lessons FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_lms_profiles_mod BEFORE UPDATE ON lms_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
