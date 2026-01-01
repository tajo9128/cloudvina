
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Play, CheckCircle, Clock, BarChart, BookOpen, Star, Share2, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../contexts/AuthContext';

export default function CourseDetailPage() {
    const { courseId } = useParams();
    const { user } = useAuth();
    const [course, setCourse] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [activeModule, setActiveModule] = useState(null);

    useEffect(() => {
        fetchCourseAndCurriculum();
    }, [courseId]);

    const fetchCourseAndCurriculum = async () => {
        setLoading(true);
        try {
            const { data, error } = await supabase
                .from('lms_courses')
                .select(`
                    *,
                    lms_profiles: instructor_id(display_name, avatar_url, bio),
                    lms_modules(
                        id, title, order_index,
                        lms_lessons(id, title, ordering: order_index, duration_minutes, type: video_url)
                    )
                `)
                .eq('id', courseId)
                .single();

            if (error) throw error;

            // Sort modules and lessons
            const sortedModules = (data.lms_modules || [])
                .sort((a, b) => a.order_index - b.order_index)
                .map(m => ({
                    ...m,
                    lms_lessons: (m.lms_lessons || []).sort((a, b) => a.ordering - b.ordering)
                }));

            setCourse({ ...data, lms_modules: sortedModules });
            if (sortedModules.length > 0) setActiveModule(sortedModules[0].id);

        } catch (err) {
            console.error('Error fetching course details:', err);
            setError('Failed to load course details. ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div className="text-center py-20 text-slate-500">Loading course...</div>;
    if (error || !course) return (
        <div className="text-center py-20 text-red-500 flex flex-col items-center">
            <AlertCircle className="w-8 h-8 mb-2" />
            <p>{error || 'Course not found'}</p>
            <Link to="/courses" className="mt-4 text-primary-600 font-bold hover:underline">Back to Courses</Link>
        </div>
    );

    const totalLessons = course.lms_modules.reduce((acc, m) => acc + m.lms_lessons.length, 0);

    return (
        <div className="bg-slate-50 min-h-screen pb-12">
            {/* Hero Section */}
            <div className="bg-slate-900 text-white py-12 lg:py-20 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-900/50"></div>
                <div className="container mx-auto px-4 relative z-10">
                    <div className="max-w-4xl">
                        <div className="flex items-center gap-2 text-primary-300 font-bold text-sm uppercase tracking-wider mb-4">
                            <span className="bg-white/10 px-2 py-1 rounded">{course.difficulty}</span>
                            <span>•</span>
                            <span>{new Date(course.created_at).toLocaleDateString()}</span>
                        </div>
                        <h1 className="text-3xl md:text-5xl font-bold mb-6 leading-tight">
                            {course.title}
                        </h1>
                        <p className="text-lg text-slate-300 mb-8 max-w-2xl">
                            {course.description}
                        </p>
                        <div className="flex flex-wrap items-center gap-6 text-sm font-medium text-slate-300 mb-8">
                            <div className="flex items-center gap-2">
                                {course.lms_profiles?.avatar_url ? (
                                    <img src={course.lms_profiles.avatar_url} alt="" className="w-8 h-8 rounded-full border border-white/20" />
                                ) : (
                                    <div className="w-8 h-8 rounded-full bg-slate-700 border border-white/20"></div>
                                )}
                                <span>Instructor: <span className="text-white hover:text-primary-300 transition-colors cursor-pointer">{course.lms_profiles?.display_name || 'Unknown'}</span></span>
                            </div>
                            <div className="flex items-center gap-2">
                                <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                                <span className="text-white">4.9/5.0</span>
                                <span>(120 ratings)</span>
                            </div>
                            <div className="flex items-center gap-1">
                                <BookOpen className="w-4 h-4" />
                                <span>{totalLessons} Lessons</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="container mx-auto px-4 -mt-10 relative z-20">
                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Main Content */}
                    <div className="lg:col-span-2 space-y-8">
                        {/* What you'll learn */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8">
                            <h3 className="text-xl font-bold text-slate-900 mb-6">Course Overview</h3>
                            <div className="prose prose-slate max-w-none text-slate-600">
                                <p>{course.description}</p>
                                {/* Placeholder for more details */}
                                <p className="mt-4">In this course, you will dive deep into the fundamentals and advanced concepts. Ideal for anyone looking to upskill in {course.title}.</p>
                            </div>
                        </div>

                        {/* Curriculum */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                            <div className="p-6 border-b border-slate-200 flex justify-between items-center">
                                <h3 className="text-xl font-bold text-slate-900">Curriculum</h3>
                                <span className="text-sm text-slate-500 font-medium">{course.lms_modules.length} Modules • {totalLessons} Lessons</span>
                            </div>
                            <div className="divide-y divide-slate-100">
                                {course.lms_modules.length === 0 ? (
                                    <div className="p-6 text-center text-slate-500 italic">No content added yet.</div>
                                ) : (
                                    course.lms_modules.map((module) => (
                                        <div key={module.id} className="bg-slate-50">
                                            <button
                                                onClick={() => setActiveModule(activeModule === module.id ? null : module.id)}
                                                className="w-full flex items-center justify-between p-4 text-left hover:bg-slate-100 transition-colors"
                                            >
                                                <div className="flex items-center gap-3">
                                                    {activeModule === module.id ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                                                    <span className="font-bold text-slate-900">{module.title}</span>
                                                </div>
                                                <span className="text-xs font-bold text-slate-500 bg-white px-2 py-1 rounded border border-slate-200">
                                                    {module.lms_lessons.length} lessons
                                                </span>
                                            </button>
                                            {activeModule === module.id && (
                                                <div className="bg-white border-t border-slate-100 divide-y divide-slate-50">
                                                    {module.lms_lessons.map((lesson) => (
                                                        <div key={lesson.id} className="flex items-center justify-between p-4 pl-12 hover:bg-slate-50 transition-colors group">
                                                            <div className="flex items-center gap-3">
                                                                <div className="w-8 h-8 rounded-full bg-primary-50 text-primary-600 flex items-center justify-center flex-shrink-0">
                                                                    <Play className="w-4 h-4 fill-current" />
                                                                </div>
                                                                <div>
                                                                    <div className="text-sm font-medium text-slate-900 group-hover:text-primary-600 transition-colors">
                                                                        {lesson.title}
                                                                    </div>
                                                                    <div className="text-xs text-slate-400">Video</div>
                                                                </div>
                                                            </div>
                                                            {/* Preview or Lock logic could go here */}
                                                            {/* If enrolled, link to lesson view */}
                                                            <Link to={`/ lesson / ${course.id}/${lesson.id}`} className="px-3 py-1 text-xs font-bold text-primary-600 bg-primary-50 rounded hover:bg-primary-100 transition-colors" >
                                                                Watch
                                                            </Link >
                                                        </div >
                                                    ))}
                                                </div >
                                            )}
                                        </div >
                                    ))
                                )}
                            </div >
                        </div >
                    </div >

                    {/* Sidebar */}
                    < div className="lg:col-span-1" >
                        <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-6 sticky top-28">
                            <div className="aspect-video rounded-lg bg-slate-900 mb-6 relative group cursor-pointer overflow-hidden">
                                {course.thumbnail_url ? (
                                    <img src={course.thumbnail_url} alt="" className="w-full h-full object-cover opacity-80 group-hover:opacity-60 transition-opacity" />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-slate-500">No Preview</div>
                                )}
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <div className="w-16 h-16 bg-white/20 backdrop-blur rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                                        <Play className="w-6 h-6 text-white fill-white ml-1" />
                                    </div>
                                </div>
                            </div>

                            <div className="mb-6">
                                <div className="flex items-end gap-2 mb-2">
                                    <span className="text-3xl font-bold text-slate-900">
                                        {course.price === 0 ? 'Free' : `$${course.price}`}
                                    </span>
                                    {course.price > 0 && <span className="text-lg text-slate-400 line-through mb-1">$99.99</span>}
                                </div>
                                <button className="w-full py-3 bg-primary-600 text-white font-bold rounded-xl shadow-lg shadow-primary-600/20 hover:bg-primary-700 hover:shadow-xl hover:translate-y-[-1px] transition-all mb-3">
                                    Enroll Now
                                </button>
                                <p className="text-center text-xs text-slate-500">30-Day Money-Back Guarantee</p>
                            </div>

                            <div className="space-y-4 pt-6 border-t border-slate-100">
                                <div className="flex items-center gap-3 text-sm text-slate-600">
                                    <Clock className="w-5 h-5 text-slate-400" />
                                    <span>Self-paced learning</span>
                                </div>
                                <div className="flex items-center gap-3 text-sm text-slate-600">
                                    <BookOpen className="w-5 h-5 text-slate-400" />
                                    <span>{totalLessons} lessons & exercises</span>
                                </div>
                                <div className="flex items-center gap-3 text-sm text-slate-600">
                                    <CheckCircle className="w-5 h-5 text-slate-400" />
                                    <span>Certificate of completion</span>
                                </div>
                                <div className="flex items-center gap-3 text-sm text-slate-600">
                                    <Share2 className="w-5 h-5 text-slate-400" />
                                    <span>Full lifetime access</span>
                                </div>
                            </div>
                        </div>
                    </div >
                </div >
            </div >
        </div >
    );
}

