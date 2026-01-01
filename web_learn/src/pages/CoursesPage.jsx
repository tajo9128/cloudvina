
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Search, Filter, BookOpen, Clock, BarChart, Star } from 'lucide-react';
import { supabase } from '../lib/supabase';

export default function CoursesPage() {
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedLevel, setSelectedLevel] = useState('All');
    const [courses, setCourses] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        fetchCourses();
    }, []);

    const fetchCourses = async () => {
        setLoading(true);
        try {
            const { data, error } = await supabase
                .from('lms_courses')
                .select(`
                    *,
                    lms_profiles: instructor_id(display_name, avatar_url),
                    lms_modules(count)
                `)
                .eq('is_published', true) // Only show published courses
                .order('created_at', { ascending: false });

            if (error) throw error;
            setCourses(data || []);
        } catch (err) {
            console.error('Error fetching courses:', err);
            setError('Failed to load courses. ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    const filteredCourses = courses.filter(course => {
        const matchesSearch = course.title.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesLevel = selectedLevel === 'All' || course.difficulty.toLowerCase() === selectedLevel.toLowerCase();
        return matchesSearch && matchesLevel;
    });

    return (
        <div className="py-12 bg-slate-50 min-h-screen">
            <div className="container mx-auto px-4">
                {/* Header */}
                <div className="text-center max-w-2xl mx-auto mb-12">
                    <h1 className="text-4xl font-bold text-slate-900 mb-4">Explore Our Courses</h1>
                    <p className="text-lg text-slate-600">Master molecular docking, drug design, and bioinformatics with our expert-led courses.</p>
                </div>

                {/* Filters */}
                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 mb-8 flex flex-col md:flex-row gap-4 justify-between items-center">
                    <div className="relative w-full md:w-96">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 w-5 h-5" />
                        <input
                            type="text"
                            placeholder="Search courses..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full pl-10 pr-4 py-3 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                        />
                    </div>
                    <div className="flex items-center gap-2 w-full md:w-auto overflow-x-auto pb-2 md:pb-0">
                        <Filter className="w-5 h-5 text-slate-400 hidden md:block" />
                        {['All', 'Beginner', 'Intermediate', 'Advanced'].map(level => (
                            <button
                                key={level}
                                onClick={() => setSelectedLevel(level)}
                                className={`px-4 py-2 rounded-lg text-sm font-bold whitespace-nowrap transition-colors ${selectedLevel === level
                                    ? 'bg-slate-900 text-white'
                                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                    } `}
                            >
                                {level}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Grid */}
                {loading ? (
                    <div className="text-center py-20 text-slate-500">Loading courses...</div>
                ) : filteredCourses.length === 0 ? (
                    <div className="text-center py-20 bg-white rounded-xl border border-slate-200">
                        <BookOpen className="w-16 h-16 text-slate-300 mx-auto mb-4" />
                        <h3 className="text-xl font-bold text-slate-900 mb-2">No courses found</h3>
                        <p className="text-slate-500">Try adjusting your search or filters.</p>
                    </div>
                ) : (
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {filteredCourses.map((course) => (
                            <Link to={`/courses/${course.id}`} key={course.id} className="bg-white rounded-xl border border-slate-200 overflow-hidden hover:shadow-lg transition-shadow group flex flex-col">
                                <div className="aspect-video bg-slate-200 relative overflow-hidden">
                                    {course.thumbnail_url ? (
                                        <img
                                            src={course.thumbnail_url}
                                            alt={course.title}
                                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                                        />
                                    ) : (
                                        <div className="w-full h-full flex items-center justify-center bg-slate-100 text-slate-400">
                                            <BookOpen className="w-12 h-12" />
                                        </div>
                                    )}
                                    <div className="absolute top-4 left-4 bg-white/90 backdrop-blur px-3 py-1 rounded-full text-xs font-bold text-slate-900 uppercase tracking-wider">
                                        {course.difficulty}
                                    </div>
                                    {course.price === 0 && (
                                        <div className="absolute top-4 right-4 bg-green-500 text-white px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider shadow-sm">
                                            Free
                                        </div>
                                    )}
                                </div>
                                <div className="p-6 flex flex-col flex-1">
                                    <h3 className="text-xl font-bold text-slate-900 mb-2 group-hover:text-primary-600 transition-colors line-clamp-2">
                                        {course.title}
                                    </h3>
                                    <p className="text-slate-600 text-sm mb-6 line-clamp-2 flex-1">
                                        {course.description || 'No description available.'}
                                    </p>

                                    <div className="flex items-center gap-4 text-xs text-slate-500 font-medium mb-6">
                                        <div className="flex items-center gap-1">
                                            <BookOpen className="w-4 h-4" />
                                            {course.lms_modules?.[0]?.count || 0} Modules
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <BarChart className="w-4 h-4" />
                                            {course.difficulty}
                                        </div>
                                    </div>

                                    <div className="border-t border-slate-100 pt-4 flex items-center justify-between mt-auto">
                                        <div className="flex items-center gap-2">
                                            <span className="text-slate-400 font-normal">({course.students})</span>
                                        </div>
                                    </div>
                                </div>
                            </Link>
                        ))}
                    </div>
                ) : (
                <div className="text-center py-20">
                    <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-400">
                        <Search className="w-8 h-8" />
                    </div>
                    <h3 className="text-xl font-bold text-slate-900 mb-2">No courses found</h3>
                    <p className="text-slate-500">Try adjusting your search or filters</p>
                </div>
                )}
            </div>
        </div>
    );
}
