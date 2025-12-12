import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Save, ArrowLeft, Plus, Trash, GripVertical, Image as ImageIcon, Loader2 } from 'lucide-react';
import { supabase } from '../../lib/supabase';
import { useAuth } from '../../contexts/AuthContext';

export default function AdminCourseEditorPage() {
    const { courseId } = useParams();
    const navigate = useNavigate();
    const { user } = useAuth();
    const isEditing = !!courseId;

    const [activeTab, setActiveTab] = useState('general');
    const [loading, setLoading] = useState(isEditing);
    const [saving, setSaving] = useState(false);

    const [formData, setFormData] = useState({
        title: '',
        description: '',
        difficulty: 'beginner',
        price: 0,
        thumbnail_url: '',
        is_published: false,
        modules: [
            { id: 'temp-m1', title: 'Introduction', lessons: [] }
        ]
    });

    useEffect(() => {
        if (isEditing) {
            fetchCourseData();
        } else {
            setLoading(false);
        }
    }, [courseId]);

    const fetchCourseData = async () => {
        try {
            const { data: course, error } = await supabase
                .from('lms_courses')
                .select(`
                    *,
                    lms_modules (
                        id, title, order_index,
                        lms_lessons (id, title, ordering:order_index, type:video_url, duration_minutes) -- simplify map
                    )
                `)
                .eq('id', courseId)
                .single();

            if (error) throw error;

            // Transform data structure to match state
            const modules = (course.lms_modules || []).sort((a, b) => a.order_index - b.order_index).map(m => ({
                id: m.id,
                title: m.title,
                lessons: (m.lms_lessons || []).sort((a, b) => a.ordering - b.ordering).map(l => ({
                    id: l.id,
                    title: l.title,
                    type: 'video',
                    duration: l.duration_minutes ? `${l.duration_minutes}:00` : '0:00',
                    duration_minutes: l.duration_minutes,
                    video_url: l.video_url
                }))
            }));

            setFormData({
                title: course.title,
                description: course.description || '',
                difficulty: course.difficulty || 'beginner',
                price: course.price || 0,
                thumbnail_url: course.thumbnail_url || '',
                is_published: course.is_published,
                modules: modules.length ? modules : [{ id: 'temp-m1', title: 'Introduction', lessons: [] }]
            });
        } catch (err) {
            console.error('Error fetching course:', err);
            alert('Failed to load course data');
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!formData.title) return alert('Course title is required');
        setSaving(true);

        try {
            // 1. Upsert Course
            const courseData = {
                title: formData.title,
                slug: formData.title.toLowerCase().replace(/[^a-z0-9]+/g, '-'), // Simple slug gen
                description: formData.description,
                difficulty: formData.difficulty,
                price: formData.price,
                thumbnail_url: formData.thumbnail_url,
                is_published: formData.is_published,
                instructor_id: user?.id
            };

            let savedCourseId = courseId;

            if (isEditing) {
                const { error } = await supabase.from('lms_courses').update(courseData).eq('id', courseId);
                if (error) throw error;
            } else {
                const { data, error } = await supabase.from('lms_courses').insert([courseData]).select().single();
                if (error) throw error;
                savedCourseId = data.id;
            }

            // 2. Handle Modules & Lessons (Simplified: Delete all and recreate, or strict upsert)
            // For MVP/Demo robustness, strict upsert is better but complex.
            // We will Iterate and Upsert modules.

            // Loop modules
            for (let i = 0; i < formData.modules.length; i++) {
                const mod = formData.modules[i];
                const modData = {
                    course_id: savedCourseId,
                    title: mod.title,
                    order_index: i
                };

                let savedModId = mod.id;
                // Check if ID is temp
                if (mod.id.toString().startsWith('temp')) {
                    const { data, error } = await supabase.from('lms_modules').insert([modData]).select().single();
                    if (error) throw error;
                    savedModId = data.id;
                } else {
                    await supabase.from('lms_modules').update(modData).eq('id', mod.id);
                }

                // Loop lessons
                for (let j = 0; j < mod.lessons.length; j++) {
                    const lesson = mod.lessons[j];
                    const lessonData = {
                        module_id: savedModId,
                        title: lesson.title,
                        slug: lesson.title.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
                        order_index: j,
                        video_url: lesson.video_url,
                        duration_minutes: parseInt(lesson.duration_minutes || 0)
                    };

                    if (lesson.id.toString().startsWith('l') || lesson.id.toString().startsWith('temp')) {
                        await supabase.from('lms_lessons').insert([lessonData]);
                    } else {
                        await supabase.from('lms_lessons').update(lessonData).eq('id', lesson.id);
                    }
                }
            }

            alert('Course saved successfully!');
            navigate('/admin/courses');
        } catch (err) {
            console.error('Error saving course:', err);
            alert('Failed to save course: ' + err.message);
        } finally {
            setSaving(false);
        }
    };

    const addModule = () => {
        setFormData({
            ...formData,
            modules: [
                ...formData.modules,
                { id: `temp-m${Date.now()}`, title: 'New Module', lessons: [] }
            ]
        });
    };

    const addLesson = (moduleIndex) => {
        const newModules = [...formData.modules];
        newModules[moduleIndex].lessons.push({
            id: `temp-l${Date.now()}`,
            title: 'New Lesson',
            type: 'video',
            duration: '0:00'
        });
        setFormData({ ...formData, modules: newModules });
    };

    const updateModuleTitle = (index, title) => {
        const newModules = [...formData.modules];
        newModules[index].title = title;
        setFormData({ ...formData, modules: newModules });
    };

    const updateLessonTitle = (moduleIndex, lessonIndex, title) => {
        const newModules = [...formData.modules];
        newModules[moduleIndex].lessons[lessonIndex].title = title;
        setFormData({ ...formData, modules: newModules });
    };

    const deleteModule = (index) => {
        // In real app, we'd delete from DB too if existing
        const newModules = [...formData.modules];
        newModules.splice(index, 1);
        setFormData({ ...formData, modules: newModules });
    };

    const deleteLesson = (moduleIndex, lessonIndex) => {
        const newModules = [...formData.modules];
        newModules[moduleIndex].lessons.splice(lessonIndex, 1);
        setFormData({ ...formData, modules: newModules });
    };

    if (loading) return <div className="p-12 text-center text-slate-500">Loading editor...</div>;

    return (
        <div className="max-w-4xl mx-auto pb-12">
            {/* Header */}
            <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-4">
                    <button onClick={() => navigate('/admin/courses')} className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                    </button>
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">
                            {isEditing ? 'Edit Course' : 'Create New Course'}
                        </h1>
                        <p className="text-slate-500 text-sm">
                            {isEditing ? `Editing "${formData.title}"` : 'Add a new course to your catalog'}
                        </p>
                    </div>
                </div>
                <button
                    onClick={handleSave}
                    disabled={saving}
                    className="flex items-center gap-2 px-6 py-2.5 bg-primary-600 text-white font-bold rounded-lg shadow-sm hover:bg-primary-700 transition-colors disabled:opacity-50"
                >
                    {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                    {saving ? 'Saving...' : 'Save Course'}
                </button>
            </div>

            {/* Tabs */}
            <div className="border-b border-slate-200 mb-8">
                <nav className="flex gap-8">
                    {['general', 'curriculum', 'settings'].map(tab => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`pb-4 px-2 font-medium text-sm border-b-2 transition-colors capitalize ${activeTab === tab
                                ? 'border-primary-600 text-primary-600'
                                : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
                                }`}
                        >
                            {tab}
                        </button>
                    ))}
                </nav>
            </div>

            {/* Content */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 sm:p-8">
                {activeTab === 'general' && (
                    <div className="space-y-6">
                        <div>
                            <label className="block text-sm font-bold text-slate-700 mb-2">Course Title</label>
                            <input
                                type="text"
                                value={formData.title}
                                onChange={e => setFormData({ ...formData, title: e.target.value })}
                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
                                placeholder="e.g. Molecular Docking 101"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-bold text-slate-700 mb-2">Description</label>
                            <textarea
                                value={formData.description}
                                onChange={e => setFormData({ ...formData, description: e.target.value })}
                                rows="4"
                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
                                placeholder="What will students learn?"
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-bold text-slate-700 mb-2">Level</label>
                                <select
                                    value={formData.difficulty}
                                    onChange={e => setFormData({ ...formData, difficulty: e.target.value })}
                                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
                                >
                                    <option value="beginner">Beginner</option>
                                    <option value="intermediate">Intermediate</option>
                                    <option value="advanced">Advanced</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-bold text-slate-700 mb-2">Price</label>
                                <input
                                    type="number"
                                    value={formData.price}
                                    onChange={e => setFormData({ ...formData, price: e.target.value })}
                                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
                                    placeholder="0"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-bold text-slate-700 mb-2">Thumbnail URL</label>
                            <input
                                type="text"
                                value={formData.thumbnail_url}
                                onChange={e => setFormData({ ...formData, thumbnail_url: e.target.value })}
                                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
                                placeholder="https://..."
                            />
                        </div>
                    </div>
                )}

                {activeTab === 'curriculum' && (
                    <div className="space-y-8">
                        {formData.modules.map((module, mIndex) => (
                            <div key={module.id} className="bg-slate-50 rounded-xl border border-slate-200 overflow-hidden">
                                <div className="p-4 bg-slate-100 border-b border-slate-200 flex items-center gap-3">
                                    <GripVertical className="w-5 h-5 text-slate-400 cursor-move" />
                                    <span className="font-bold text-slate-500 text-sm uppercase tracking-wider">Module {mIndex + 1}:</span>
                                    <input
                                        type="text"
                                        value={module.title}
                                        onChange={e => updateModuleTitle(mIndex, e.target.value)}
                                        className="bg-transparent font-bold text-slate-900 outline-none flex-1 focus:bg-white focus:px-2 focus:py-1 rounded"
                                    />
                                    <button onClick={() => deleteModule(mIndex)} className="text-slate-400 hover:text-red-500 p-2">
                                        <Trash className="w-4 h-4" />
                                    </button>
                                </div>
                                <div className="p-4 space-y-3">
                                    {module.lessons.map((lesson, lIndex) => (
                                        <div key={lesson.id} className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm space-y-3">
                                            <div className="flex items-center gap-3">
                                                <GripVertical className="w-4 h-4 text-slate-300 cursor-move" />
                                                <div className="text-xs font-bold text-slate-400 w-6">{lIndex + 1}.</div>
                                                <input
                                                    type="text"
                                                    value={lesson.title}
                                                    onChange={e => updateLessonTitle(mIndex, lIndex, e.target.value)}
                                                    className="flex-1 text-sm font-bold text-slate-700 outline-none border-b border-transparent focus:border-primary-300"
                                                    placeholder="Lesson Title"
                                                />
                                                <button onClick={() => deleteLesson(mIndex, lIndex)} className="text-slate-300 hover:text-red-500">
                                                    <X className="w-4 h-4" />
                                                </button>
                                            </div>

                                            {/* Lesson Content Fields */}
                                            <div className="pl-12 grid grid-cols-1 sm:grid-cols-2 gap-4">
                                                <div>
                                                    <label className="block text-xs font-bold text-slate-500 mb-1">Video URL (Embed/MP4)</label>
                                                    <input
                                                        type="text"
                                                        value={lesson.video_url || ''}
                                                        onChange={e => {
                                                            const newModules = [...formData.modules];
                                                            newModules[mIndex].lessons[lIndex].video_url = e.target.value;
                                                            setFormData({ ...formData, modules: newModules });
                                                        }}
                                                        className="w-full px-3 py-1.5 text-xs border border-slate-200 rounded focus:border-primary-500 outline-none"
                                                        placeholder="https://..."
                                                    />
                                                </div>
                                                <div>
                                                    <label className="block text-xs font-bold text-slate-500 mb-1">Duration (Min)</label>
                                                    <input
                                                        type="number"
                                                        value={lesson.duration_minutes || ''}
                                                        onChange={e => {
                                                            const newModules = [...formData.modules];
                                                            newModules[mIndex].lessons[lIndex].duration_minutes = e.target.value;
                                                            // Also update display duration for UI if needed, or just rely on minutes
                                                            newModules[mIndex].lessons[lIndex].duration = `${e.target.value}:00`;
                                                            setFormData({ ...formData, modules: newModules });
                                                        }}
                                                        className="w-full px-3 py-1.5 text-xs border border-slate-200 rounded focus:border-primary-500 outline-none"
                                                        placeholder="e.g. 15"
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                    <button
                                        onClick={() => addLesson(mIndex)}
                                        className="w-full py-2 border-2 border-dashed border-slate-200 rounded-lg text-sm font-bold text-slate-400 hover:border-primary-300 hover:text-primary-600 hover:bg-primary-50 transition-all flex justify-center items-center gap-2"
                                    >
                                        <Plus className="w-4 h-4" /> Add Lesson
                                    </button>
                                </div>
                            </div>
                        ))}

                        <button
                            onClick={addModule}
                            className="w-full py-4 bg-slate-800 text-white font-bold rounded-xl hover:bg-slate-700 transition-colors flex justify-center items-center gap-2"
                        >
                            <Plus className="w-5 h-5" /> Add New Module
                        </button>
                    </div>
                )}

                {activeTab === 'settings' && (
                    <div className="text-center py-12 text-slate-500">
                        Course settings like SEO, enrollment duration, and certificates configuration will go here.
                        <div className="mt-4">
                            <label className="flex items-center justify-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={formData.is_published}
                                    onChange={e => setFormData({ ...formData, is_published: e.target.checked })}
                                    className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
                                />
                                <span className="font-bold text-slate-700">Publish Course</span>
                            </label>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// Helper icon component
function X({ className }) {
    return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
    );
}
