import React, { useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Download, ChevronLeft, Award, Share2 } from 'lucide-react';
import { courses } from '../data/courseData';
import { useProgressStore } from '../stores/useProgressStore';

export default function CertificatePage() {
    const { slug } = useParams();
    const certificateRef = useRef(null);
    const course = courses.find(c => c.slug === slug);

    // Get Progress
    const getCourseProgress = useProgressStore(state => state.getCourseProgress);

    // Calculate total lessons
    let totalLessons = 0;
    if (course) {
        course.curriculum.forEach(mod => {
            totalLessons += mod.lessons.length;
        });
    }

    const progress = course ? getCourseProgress(course.id, totalLessons) : 0;
    const isComplete = progress === 100;

    const handlePrint = () => {
        window.print();
    };

    if (!course) return <div>Course not found</div>;

    if (!isComplete) {
        return (
            <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
                <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 max-w-md text-center">
                    <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-6">
                        <Award className="w-8 h-8 text-slate-400" />
                    </div>
                    <h1 className="text-2xl font-bold text-slate-900 mb-2">Certificate Locked</h1>
                    <p className="text-slate-600 mb-6">
                        You need to complete 100% of the course to unlock your certificate. You are currently at <span className="font-bold text-primary-600">{progress}%</span>.
                    </p>
                    <Link to={`/courses/${slug}`} className="px-6 py-3 bg-primary-600 text-white font-bold rounded-lg hover:bg-primary-700 transition-colors block">
                        Keep Learning
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-900 flex flex-col items-center py-12 px-4">

            {/* Action Bar */}
            <div className="w-full max-w-5xl flex justify-between items-center mb-8 px-4 text-white">
                <Link to={`/courses/${slug}`} className="flex items-center gap-2 hover:text-primary-400 transition-colors">
                    <ChevronLeft className="w-5 h-5" /> Back to Course
                </Link>
                <div className="flex gap-4">
                    <button className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors border border-slate-700">
                        <Share2 className="w-4 h-4" /> Share
                    </button>
                    <button
                        onClick={handlePrint}
                        className="flex items-center gap-2 px-6 py-2 bg-primary-600 hover:bg-primary-500 rounded-lg font-bold shadow-lg shadow-primary-500/20 transition-all transform hover:scale-105"
                    >
                        <Download className="w-4 h-4" /> Download PDF
                    </button>
                </div>
            </div>

            {/* Certificate Container */}
            <div className="w-full max-w-5xl bg-white p-2 shadow-2xl rounded-xl overflow-hidden print:p-0 print:shadow-none print:w-full print:max-w-none">
                <div
                    ref={certificateRef}
                    className="relative w-full aspect-[1.414/1] bg-white border-8 border-double border-slate-900 p-12 flex flex-col items-center justify-center text-center print:border-none print:w-full print:h-screen"
                    style={{ backgroundImage: 'radial-gradient(circle at 50% 50%, #fafafa 0%, #ffffff 100%)' }}
                >
                    {/* Background Pattern */}
                    <div className="absolute inset-0 opacity-[0.03] pointer-events-none"
                        style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000000' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")` }}
                    ></div>

                    {/* Logo */}
                    <div className="mb-12">
                        <div className="flex items-center gap-3 justify-center mb-2">
                            <div className="w-12 h-12 bg-primary-600 rounded-xl flex items-center justify-center text-white">
                                <Award className="w-8 h-8" />
                            </div>
                            <span className="text-3xl font-bold font-display text-slate-900 tracking-tight">
                                bio<span className="text-primary-600">Dockify</span> Learn
                            </span>
                        </div>
                        <div className="text-xs font-bold uppercase tracking-[0.3em] text-slate-400">Certificate of Completion</div>
                    </div>

                    {/* Content */}
                    <div className="space-y-8 max-w-3xl relative z-10">
                        <p className="text-xl text-slate-600 font-serif italic">This certifies that</p>

                        <h2 className="text-5xl md:text-6xl font-bold text-slate-900 border-b-2 border-slate-900 pb-8 px-12 inline-block font-display">
                            Guest User
                        </h2>

                        <p className="text-xl text-slate-600 font-serif italic mt-8">
                            has successfully completed the course
                        </p>

                        <h3 className="text-3xl md:text-4xl font-bold text-primary-900 mt-4 leading-tight">
                            {course.title}
                        </h3>

                        <div className="flex justify-center gap-16 mt-16 pt-16">
                            <div className="text-center">
                                <div className="w-48 border-b border-slate-400 mb-2"></div>
                                <div className="font-bold text-slate-900">Dr. Sarah Smith</div>
                                <div className="text-xs uppercase tracking-wider text-slate-500">Lead Instructor</div>
                            </div>
                            <div className="text-center">
                                <div className="w-48 border-b border-slate-400 mb-2 font-display text-slate-900">{new Date().toLocaleDateString()}</div>
                                <div className="text-xs uppercase tracking-wider text-slate-500">Date Issued</div>
                            </div>
                        </div>

                        <div className="text-xs text-slate-400 mt-12">
                            Certificate ID: {Math.random().toString(36).substr(2, 9).toUpperCase()} • Verify at learn.biodockify.com/verify
                        </div>
                    </div>
                </div>
            </div>

            {/* Print Styles */}
            <style>
                {`
                    @media print {
                        body * {
                            visibility: hidden;
                        }
                        .print\\:border-none {
                            border: none !important;
                        }
                        .print\\:w-full {
                            width: 100% !important;
                            max-width: none !important;
                        }
                        .print\\:h-screen {
                            height: 100vh !important;
                        }
                        .print\\:p-0 {
                            padding: 0 !important;
                        }
                        .print\\:shadow-none {
                            box-shadow: none !important;
                        }
                        div[class*="CertificateContainer"] * {
                            visibility: visible;
                        }
                        div[class*="CertificateContainer"] {
                            position: absolute;
                            left: 0;
                            top: 0;
                            width: 100%;
                            height: 100%;
                            margin: 0;
                            padding: 2cm; 
                        }
                    }
                `}
            </style>
        </div>
    );
}
