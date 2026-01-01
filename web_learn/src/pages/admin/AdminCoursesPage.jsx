import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Plus, Search, MoreVertical, Edit, Trash, Eye, BookOpen, AlertCircle } from 'lucide-react';
import { supabase } from '../../lib/supabase';

export default function AdminCoursesPage() {
    const [searchTerm, setSearchTerm] = useState('');
    const [courses, setCourses] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        fetchCourses();
    }, []);

    const fetchCourses = async () => {
        setLoading(true);
        try {
            // Fetch courses with instructor details and module/lesson counts
            // Note: Counts might need a separate query or a view in a real complex app,
            // but for now we'll fetch raw and filter/count or just show basics.
            // Supabase generic select with relations:
            const { data, error } = await supabase
                .from('lms_courses')
                .select(`
    *,
    lms_profiles: instructor_id(display_name, avatar_url),
        lms_modules(count)
            `)
                .order('created_at', { ascending: false });

            if (error) throw error;
            setCourses(data || []);
        } catch (err) {
            console.error(err);
            setError('Failed to fetch courses. ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    const filteredCourses = courses.filter(course =>
        course.title.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-slate-900">Courses</h1>
                    <p className="text-slate-500">Manage your course catalog and curriculum (Real Database).</p>
                </div>
                <Link to="/admin/courses/new" className="px-4 py-2 bg-primary-600 text-white font-bold rounded-lg shadow-sm hover:bg-primary-700 transition-colors flex items-center gap-2">
                    <Plus className="w-5 h-5" /> Create Course
                </Link>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-4 border-b border-slate-200">
                    <div className="relative max-w-md">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 w-5 h-5" />
                        <input
                            type="text"
                            placeholder="Search courses..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                        />
                    </div>
                </div>

                <div className="overflow-x-auto">
                    {loading ? (
                        <div className="p-8 text-center text-slate-500">Loading courses...</div>
                    ) : error ? (
                        <div className="p-8 text-center text-red-500 flex flex-col items-center gap-2">
                            <AlertCircle className="w-6 h-6" />
                            {error}
                        </div>
                    ) : filteredCourses.length === 0 ? (
                        <div className="p-12 text-center">
                            <BookOpen className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                            <h3 className="text-lg font-medium text-slate-900">No courses yet</h3>
                            <p className="text-slate-500 mb-6">Get started by creating your first course.</p>
                            <Link to="/admin/courses/new" className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white font-bold rounded-lg hover:bg-primary-700 transition-colors">
                                <Plus className="w-4 h-4" /> Create Course
                            </Link>
                        </div>
                    ) : (
                        <table className="w-full text-left text-sm">
                            <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                                <tr>
                                    <th className="px-6 py-4">Title</th>
                                    <th className="px-6 py-4">Instructor</th>
                                    <th className="px-6 py-4">Level</th>
                                    {/* <th className="px-6 py-4">Students</th> */}
                                    <th className="px-6 py-4">Status</th>
                                    <th className="px-6 py-4 text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {filteredCourses.map((course) => (
                                    <tr key={course.id} className="hover:bg-slate-50 transition-colors group">
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded bg-slate-100 overflow-hidden flex-shrink-0">
                                                    {course.thumbnail_url ? (
                                                        <img src={course.thumbnail_url} alt="" className="w-full h-full object-cover" />
                                                    ) : (
                                                        <div className="w-full h-full flex items-center justify-center bg-slate-200 text-slate-400">
                                                            <BookOpen className="w-5 h-5" />
                                                        </div>
                                                    )}
                                                </div>
                                                <div>
                                                    <div className="font-bold text-slate-900 group-hover:text-primary-600 transition-colors">
                                                        {course.title}
                                                    </div>
                                                    <div className="text-xs text-slate-500">
                                                        {course.lms_modules?.[0]?.count || 0} Modules
                                                    </div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-2">
                                                {course.lms_profiles?.avatar_url ? (
                                                    <img src={course.lms_profiles.avatar_url} alt="" className="w-6 h-6 rounded-full" />
                                                ) : (
                                                    <div className="w-6 h-6 rounded-full bg-slate-300"></div>
                                                )}
                                                <span className="text-slate-700">{course.lms_profiles?.display_name || 'Unknown'}</span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`px - 2 py - 1 rounded text - xs font - bold uppercase tracking - wider ${course.difficulty === 'beginner' ? 'bg-green-100 text-green-700' :
                                                course.difficulty === 'intermediate' ? 'bg-yellow-100 text-yellow-700' :
                                                    'bg-red-100 text-red-700'
                                                } `}>
                                                {course.difficulty || 'All Levels'}
                                            </span>
                                        </td>
                                        {/* Subquery for student count is expensive, skipping for now */}
                                        <td className="px-6 py-4">
                                            {course.is_published ? (
                                                <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-bold rounded-full">
                                                    Published
                                                </span>
                                            ) : (
                                                <span className="px-2 py-1 bg-slate-100 text-slate-500 text-xs font-bold rounded-full">
                                                    Draft
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <div className="flex justify-end gap-2">
                                                <button className="p-2 text-slate-400 hover:text-primary-600 hover:bg-primary-50 rounded transition-colors" title="View">
                                                    <Eye className="w-4 h-4" />
                                                </button>
                                                <Link to={`/ admin / courses / ${course.id}/edit`} className="p-2 text-slate-400 hover:text-amber-600 hover:bg-amber-50 rounded transition-colors" title="Edit" >
                                                    <Edit className="w-4 h-4" />
                                                </Link >
                                                <button className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors" title="Delete">
                                                    <Trash className="w-4 h-4" />
                                                </button>
                                            </div >
                                        </td >
                                    </tr >
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            </div >
        </div >
    );
}
