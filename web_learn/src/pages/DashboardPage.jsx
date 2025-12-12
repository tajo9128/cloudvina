import React from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, Award, Clock, TrendingUp, PlayCircle, MoreVertical } from 'lucide-react';
import { courses } from '../data/courseData';
import { useProgressStore } from '../stores/useProgressStore';

export default function DashboardPage() {
    // In a real app, we'd get the user from AuthContext
    const user = {
        name: 'Guest User',
        avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Felix',
        role: 'Student'
    };

    const getCourseProgress = useProgressStore(state => state.getCourseProgress);

    // Calculate stats based on mock enrollment (assuming enrolled in all for demo)
    const enrolledCourses = courses.map(course => {
        let totalLessons = 0;
        course.curriculum.forEach(mod => totalLessons += mod.lessons.length);
        const progress = getCourseProgress(course.id, totalLessons);
        return { ...course, progress, totalLessons };
    });

    const inProgress = enrolledCourses.filter(c => c.progress > 0 && c.progress < 100).length;
    const completed = enrolledCourses.filter(c => c.progress === 100).length;
    const totalHours = 12; // Mock total learning hours

    return (
        <div className="min-h-screen bg-slate-50 py-8">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

                {/* Welcome Section */}
                <div className="flex flex-col md:flex-row items-center justify-between mb-8 gap-4">
                    <div className="flex items-center gap-4 w-full md:w-auto">
                        <img src={user.avatar} alt="Profile" className="w-16 h-16 rounded-full bg-white p-1 border border-slate-200" />
                        <div>
                            <h1 className="text-2xl font-bold text-slate-900">Welcome back, {user.name}!</h1>
                            <p className="text-slate-500">You've learned for <span className="text-primary-600 font-bold">{totalHours} hours</span> this week.</p>
                        </div>
                    </div>
                    <div className="flex gap-3 w-full md:w-auto">
                        <Link to="/courses" className="px-5 py-2.5 bg-primary-600 text-white font-bold rounded-lg hover:bg-primary-700 transition-colors shadow-sm flex items-center justify-center gap-2 flex-1 md:flex-none">
                            <BookOpen className="w-4 h-4" /> Browse Courses
                        </Link>
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex items-center gap-4">
                        <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center">
                            <BookOpen className="w-6 h-6" />
                        </div>
                        <div>
                            <div className="text-3xl font-bold text-slate-900">{inProgress}</div>
                            <div className="text-sm font-medium text-slate-500">Courses in Progress</div>
                        </div>
                    </div>
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex items-center gap-4">
                        <div className="w-12 h-12 bg-green-50 text-green-600 rounded-xl flex items-center justify-center">
                            <Award className="w-6 h-6" />
                        </div>
                        <div>
                            <div className="text-3xl font-bold text-slate-900">{completed}</div>
                            <div className="text-sm font-medium text-slate-500">Certificates Earned</div>
                        </div>
                    </div>
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex items-center gap-4">
                        <div className="w-12 h-12 bg-purple-50 text-purple-600 rounded-xl flex items-center justify-center">
                            <TrendingUp className="w-6 h-6" />
                        </div>
                        <div>
                            <div className="text-3xl font-bold text-slate-900">Top 10%</div>
                            <div className="text-sm font-medium text-slate-500">Community Rank</div>
                        </div>
                    </div>
                </div>

                <div className="flex flex-col lg:flex-row gap-8">

                    {/* My Learning (Left Column) */}
                    <div className="lg:w-2/3 space-y-8">
                        <div>
                            <h2 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                                <PlayCircle className="w-5 h-5 text-primary-600" /> My Learning
                            </h2>
                            <div className="space-y-4">
                                {enrolledCourses.map(course => (
                                    <div key={course.id} className="bg-white rounded-xl border border-slate-200 p-4 flex flex-col sm:flex-row gap-6 hover:shadow-md transition-shadow group">
                                        <div className="w-full sm:w-48 h-32 rounded-lg overflow-hidden flex-shrink-0 relative">
                                            <img src={course.thumbnail} alt={course.title} className="w-full h-full object-cover" />
                                            <div className="absolute inset-0 bg-black/10 group-hover:bg-black/0 transition-colors"></div>
                                        </div>
                                        <div className="flex-1 flex flex-col justify-center">
                                            <div className="flex justify-between items-start mb-2">
                                                <h3 className="font-bold text-slate-900 text-lg group-hover:text-primary-600 transition-colors">
                                                    <Link to={`/courses/${course.slug}`}>{course.title}</Link>
                                                </h3>
                                                <button className="text-slate-400 hover:text-slate-600">
                                                    <MoreVertical className="w-5 h-5" />
                                                </button>
                                            </div>
                                            <div className="flex items-center gap-4 text-xs font-semibold text-slate-500 mb-4">
                                                <span>{course.lessonsCount} Lessons</span>
                                                <span>•</span>
                                                <span>{course.level}</span>
                                            </div>

                                            {/* Progress Bar */}
                                            <div className="space-y-2">
                                                <div className="flex justify-between text-xs font-bold">
                                                    <span className={course.progress === 100 ? "text-green-600" : "text-slate-700"}>
                                                        {course.progress === 100 ? 'Completed' : `${course.progress}% Complete`}
                                                    </span>
                                                    {course.progress === 100 && (
                                                        <Link to={`/courses/${course.slug}/certificate`} className="text-primary-600 hover:underline">
                                                            View Certificate
                                                        </Link>
                                                    )}
                                                </div>
                                                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                                                    <div
                                                        className={`h-full rounded-full transition-all duration-500 ${course.progress === 100 ? 'bg-green-500' : 'bg-primary-600'}`}
                                                        style={{ width: `${course.progress}%` }}
                                                    ></div>
                                                </div>
                                                {course.progress < 100 && (
                                                    <Link
                                                        to={`/courses/${course.slug}`}
                                                        className="inline-block mt-2 text-sm font-bold text-primary-600 hover:text-primary-700"
                                                    >
                                                        Continue Learning →
                                                    </Link>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Sidebar (Right Column) */}
                    <div className="lg:w-1/3 space-y-6">
                        {/* Daily Goal */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-4">Daily Goal</h3>
                            <div className="flex items-center gap-4 mb-4">
                                <div className="w-12 h-12 rounded-full border-4 border-primary-600 flex items-center justify-center font-bold text-primary-600 text-lg">
                                    2/3
                                </div>
                                <div>
                                    <div className="text-sm font-medium text-slate-900">Lessons Completed</div>
                                    <div className="text-xs text-slate-500">Keep 'em coming! 🔥</div>
                                </div>
                            </div>
                        </div>

                        {/* Achievements */}
                        <div className="bg-white rounded-xl border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-4">Recent Achievements</h3>
                            <div className="grid grid-cols-4 gap-2">
                                <div title="First Lesson" className="aspect-square bg-yellow-100 rounded-lg flex items-center justify-center text-2xl cursor-help">🏁</div>
                                <div title="Fast Learner" className="aspect-square bg-blue-100 rounded-lg flex items-center justify-center text-2xl cursor-help">⚡</div>
                                <div title="Quiz Master" className="aspect-square bg-purple-100 rounded-lg flex items-center justify-center text-2xl cursor-help">🧠</div>
                                <div className="aspect-square bg-slate-100 rounded-lg flex items-center justify-center text-slate-400 text-sm font-bold border-2 border-dashed border-slate-300">
                                    +3
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
}
