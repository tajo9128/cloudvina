import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, PlayCircle, CheckCircle, Circle, Menu, X, FileText, Download, MessageSquare, CheckSquare, Loader2 } from 'lucide-react';
import { supabase } from '../../lib/supabase';
import { useProgressStore } from '../stores/useProgressStore';
import ReactMarkdown from 'react-markdown'; // Ensure this is installed or handle simple text if not

export default function LessonPage() {
    const { slug, lessonId } = useParams();
    const navigate = useNavigate();
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [activeTab, setActiveTab] = useState('content');

    // Data State
    const [course, setCourse] = useState(null);
    const [currentLesson, setCurrentLesson] = useState(null);
    const [currentModule, setCurrentModule] = useState(null);
    const [flatLessonList, setFlatLessonList] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Store Hooks
    const markLessonComplete = useProgressStore(state => state.markLessonComplete);
    const isLessonComplete = useProgressStore(state => state.isLessonComplete);

    // Fetch Data
    useEffect(() => {
        const fetchLessonData = async () => {
            try {
                setLoading(true);

                // 1. Fetch Course with Modules and Lessons
                // We fetch everything to build the sidebar navigation
                const { data: courseData, error: courseError } = await supabase
                    .from('lms_courses')
                    .select(`
                        *,
                        modules:lms_modules(
                            *,
                            lessons:lms_lessons(*)
                        )
                    `)
                    .eq('slug', slug)
                    .single();

                if (courseError) throw courseError;
                if (!courseData) throw new Error('Course not found');

                // Sort modules and lessons by order_index
                const sortedModules = (courseData.modules || []).sort((a, b) => a.order_index - b.order_index);
                sortedModules.forEach(mod => {
                    mod.lessons = (mod.lessons || []).sort((a, b) => a.order_index - b.order_index);
                });

                // Flatten lessons for navigation
                const flatList = [];
                let foundLesson = null;
                let foundModule = null;

                sortedModules.forEach(mod => {
                    mod.lessons.forEach(lesson => {
                        flatList.push({ ...lesson, moduleId: mod.id, moduleTitle: mod.title });
                        if (lesson.id === lessonId) {
                            foundLesson = lesson;
                            foundModule = mod;
                        }
                    });
                });

                setCourse({ ...courseData, curriculum: sortedModules }); // Map to expected structure
                setFlatLessonList(flatList);
                setCurrentLesson(foundLesson);
                setCurrentModule(foundModule);

            } catch (err) {
                console.error('Error fetching lesson:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        if (slug) {
            fetchLessonData();
        }
    }, [slug, lessonId]);


    // Navigation Logic
    const currentIndex = flatLessonList.findIndex(l => l.id === lessonId);
    const prevLesson = currentIndex > 0 ? flatLessonList[currentIndex - 1] : null;
    const nextLesson = currentIndex < flatLessonList.length - 1 ? flatLessonList[currentIndex + 1] : null;

    useEffect(() => {
        // Scroll to top on lesson change
        window.scrollTo(0, 0);
    }, [lessonId]);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-slate-50">
                <Loader2 className="w-10 h-10 text-indigo-600 animate-spin" />
            </div>
        );
    }

    if (error || !course || !currentLesson) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-slate-50">
                <div className="text-center">
                    <h2 className="text-2xl font-bold text-slate-800 mb-2">Lesson Not Found</h2>
                    <p className="text-slate-500 mb-6">{error || "The requested lesson could not be loaded."}</p>
                    <Link to="/courses" className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 transition-colors">
                        Return to Catalog
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-screen bg-white overflow-hidden">

            {/* Sidebar (Curriculum) */}
            <div className={`fixed inset-y-0 left-0 z-30 w-80 bg-slate-50 border-r border-slate-200 transform transition-transform duration-300 ease-in-out ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:relative lg:translate-x-0`}>
                <div className="flex flex-col h-full">
                    {/* Sidebar Header */}
                    <div className="p-4 border-b border-slate-200">
                        <Link to={`/courses/${course.id}`} className="flex items-center gap-2 text-sm text-slate-500 hover:text-primary-600 mb-3 transition-colors">
                            <ChevronLeft className="w-4 h-4" />
                            Back to Course Home
                        </Link>
                        <h2 className="font-bold text-slate-900 line-clamp-2">{course.title}</h2>
                        {/* Progress Bar Placeholder (Integration Pending) */}
                        <div className="mt-2 w-full bg-slate-200 rounded-full h-1.5">
                            <div className="bg-green-500 h-1.5 rounded-full w-1/4"></div>
                        </div>
                        <div className="mt-1 text-xs text-slate-500 flex justify-between">
                            <span>25% Complete</span>
                            <span>{flatLessonList.length} Lessons</span>
                        </div>
                    </div>

                    {/* Module List */}
                    <div className="flex-1 overflow-y-auto">
                        {course.curriculum.map((module, mIdx) => (
                            <div key={module.id} className="border-b border-slate-100">
                                <div className="px-4 py-3 bg-slate-100/50">
                                    <h3 className="text-xs font-bold uppercase text-slate-500 tracking-wider mb-1">Module {mIdx + 1}</h3>
                                    <div className="text-sm font-semibold text-slate-900">{module.title}</div>
                                </div>
                                <div>
                                    {module.lessons.map((lesson) => {
                                        const isCompleted = isLessonComplete(course.id, lesson.id);
                                        return (
                                            <Link
                                                key={lesson.id}
                                                to={`/lesson/${slug}/${lesson.id}`}
                                                className={`flex items-start gap-3 px-4 py-3 hover:bg-slate-50 transition-colors border-l-4 ${lesson.id === lessonId ? 'border-primary-600 bg-white shadow-sm' : 'border-transparent'}`}
                                            >
                                                <div className="mt-0.5">
                                                    {lesson.id === lessonId ? (
                                                        <PlayCircle className="w-4 h-4 text-primary-600" />
                                                    ) : isCompleted ? (
                                                        <CheckCircle className="w-4 h-4 text-green-500" />
                                                    ) : (
                                                        <Circle className="w-4 h-4 text-slate-300" />
                                                    )}
                                                </div>
                                                <div>
                                                    <div className={`text-sm ${lesson.id === lessonId ? 'font-bold text-primary-700' : 'text-slate-600'}`}>
                                                        {lesson.title}
                                                    </div>
                                                    <div className="text-xs text-slate-400 mt-1">{lesson.duration_minutes ? `${lesson.duration_minutes} min` : '5 min'}</div>
                                                </div>
                                            </Link>
                                        )
                                    })}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col h-full overflow-hidden relative">

                {/* Mobile Toggle */}
                <button
                    onClick={() => setSidebarOpen(!sidebarOpen)}
                    className="lg:hidden absolute top-4 left-4 z-40 p-2 bg-white rounded-md shadow-md"
                >
                    {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                </button>

                {/* Top Bar (Lesson Nav) */}
                <div className="bg-white border-b border-slate-200 px-6 py-3 flex justify-between items-center shadow-sm z-20">
                    <h1 className="text-lg font-bold text-slate-800 hidden md:block">
                        {currentLesson.title}
                    </h1>
                    <div className="flex gap-3 ml-auto">
                        <button
                            onClick={() => prevLesson && navigate(`/lesson/${slug}/${prevLesson.id}`)}
                            disabled={!prevLesson}
                            className={`px-4 py-2 flex items-center gap-2 rounded-lg text-sm font-medium transition-colors ${!prevLesson ? 'text-slate-300 cursor-not-allowed' : 'text-slate-700 hover:bg-slate-100 border border-slate-200'}`}
                        >
                            <ChevronLeft className="w-4 h-4" /> Previous
                        </button>
                        <button
                            onClick={() => nextLesson && navigate(`/lesson/${slug}/${nextLesson.id}`)}
                            disabled={!nextLesson}
                            className={`px-4 py-2 flex items-center gap-2 rounded-lg text-sm font-medium transition-colors ${!nextLesson ? 'text-slate-300 cursor-not-allowed' : 'bg-primary-600 text-white hover:bg-primary-700 shadow-sm'}`}
                        >
                            Next <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Scrollable Content Area */}
                <div className="flex-1 overflow-y-auto bg-slate-50 p-4 md:p-8">
                    <div className="max-w-4xl mx-auto space-y-8">

                        {/* Video Player Placeholder or Iframe */}
                        <div className="aspect-video bg-black rounded-xl overflow-hidden shadow-lg relative group">
                            {currentLesson.video_url ? (
                                <iframe
                                    src={currentLesson.video_url}
                                    title={currentLesson.title}
                                    className="w-full h-full"
                                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                    allowFullScreen
                                ></iframe>
                            ) : (
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                                    <PlayCircle className="w-20 h-20 opacity-50 mb-2" />
                                    <p>No video available for this lesson.</p>
                                </div>
                            )}
                        </div>

                        <div className="flex justify-between items-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                            <div>
                                <h2 className="text-lg font-bold text-slate-900 mb-1">{currentLesson.title}</h2>
                                <p className="text-sm text-slate-500">From module: {currentModule?.title}</p>
                            </div>
                            <button
                                onClick={() => markLessonComplete(course.id, currentLesson.id)}
                                className={`px-6 py-3 rounded-lg font-bold flex items-center gap-2 transition-all ${isLessonComplete(course.id, currentLesson.id) ? 'bg-green-100 text-green-700 cursor-default' : 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg shadow-primary-500/30'}`}
                            >
                                {isLessonComplete(course.id, currentLesson.id) ? (
                                    <>
                                        <CheckCircle className="w-5 h-5" /> Completed
                                    </>
                                ) : (
                                    <>
                                        <CheckSquare className="w-5 h-5" /> Mark Complete
                                    </>
                                )}
                            </button>
                        </div>

                        {/* Tabs */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                            <div className="flex border-b border-slate-200">
                                <button
                                    onClick={() => setActiveTab('content')}
                                    className={`flex-1 py-4 text-sm font-bold text-center border-b-2 transition-colors ${activeTab === 'content' ? 'border-primary-600 text-primary-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
                                >
                                    <span className="flex items-center justify-center gap-2">
                                        <FileText className="w-4 h-4" /> Lesson Content
                                    </span>
                                </button>
                                <button
                                    onClick={() => setActiveTab('materials')}
                                    className={`flex-1 py-4 text-sm font-bold text-center border-b-2 transition-colors ${activeTab === 'materials' ? 'border-primary-600 text-primary-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
                                >
                                    <span className="flex items-center justify-center gap-2">
                                        <Download className="w-4 h-4" /> Materials
                                    </span>
                                </button>
                                <button
                                    onClick={() => setActiveTab('discussion')}
                                    className={`flex-1 py-4 text-sm font-bold text-center border-b-2 transition-colors ${activeTab === 'discussion' ? 'border-primary-600 text-primary-600' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
                                >
                                    <span className="flex items-center justify-center gap-2">
                                        <MessageSquare className="w-4 h-4" /> Discussion
                                    </span>
                                </button>
                            </div>

                            <div className="p-8">
                                {activeTab === 'content' && (
                                    <div className="prose prose-slate max-w-none">
                                        {currentLesson.content ? (
                                            <ReactMarkdown>{currentLesson.content}</ReactMarkdown>
                                        ) : (
                                            <p className="text-slate-500 italic">No additional content provided for this lesson.</p>
                                        )}
                                    </div>
                                )}
                                {activeTab === 'materials' && (
                                    <div className="text-center py-8 text-slate-500">
                                        <Download className="w-12 h-12 mx-auto text-slate-300 mb-3" />
                                        <p>No materials attached to this lesson.</p>
                                    </div>
                                )}
                                {activeTab === 'discussion' && (
                                    <div className="text-center py-8 text-slate-500">
                                        <MessageSquare className="w-12 h-12 mx-auto text-slate-300 mb-3" />
                                        <p>No questions yet. Be the first to ask!</p>
                                        <button className="mt-4 px-4 py-2 bg-slate-100 text-slate-700 font-bold rounded-lg hover:bg-slate-200 transition-colors">
                                            Ask a Question
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>

                    </div>
                </div>

            </div>
        </div>
    );
}
