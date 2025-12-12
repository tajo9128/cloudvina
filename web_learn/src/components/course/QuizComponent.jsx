import React, { useState } from 'react';
import { CheckCircle, XCircle, AlertCircle, RefreshCw, ChevronRight } from 'lucide-react';
import { useProgressStore } from '../stores/useProgressStore';

export default function QuizComponent({ lesson, courseId }) {
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [selectedOption, setSelectedOption] = useState(null);
    const [score, setScore] = useState(0);
    const [showResult, setShowResult] = useState(false);
    const [quizStarted, setQuizStarted] = useState(false);

    const markLessonComplete = useProgressStore(state => state.markLessonComplete);

    // Mock Questions (In a real app, these would come from the lesson data)
    const questions = lesson.questions || [
        {
            questionText: "What is the primary function of AutoDock Vina?",
            options: [
                { id: 1, text: "Molecular Dynamics Simulation", isCorrect: false },
                { id: 2, text: "Molecular Docking and Virtual Screening", isCorrect: true },
                { id: 3, text: "Protein Structure Prediction", isCorrect: false },
                { id: 4, text: "Genomic Sequencing", isCorrect: false }
            ]
        },
        {
            questionText: "Which file format is commonly used for the receptor in Vina?",
            options: [
                { id: 1, text: ".pdbqt", isCorrect: true },
                { id: 2, text: ".sdf", isCorrect: false },
                { id: 3, text: ".fasta", isCorrect: false },
                { id: 4, text: ".mol2", isCorrect: false }
            ]
        },
        {
            questionText: "What defines the search space in molecular docking?",
            options: [
                { id: 1, text: "The RMSD value", isCorrect: false },
                { id: 2, text: "The Grid Box", isCorrect: true },
                { id: 3, text: "The Torsional degrees of freedom", isCorrect: false },
                { id: 4, text: "The Binding Affinity", isCorrect: false }
            ]
        }
    ];

    const handleOptionSelect = (optionId) => {
        setSelectedOption(optionId);
    };

    const handleNextQuestion = () => {
        const isCorrect = questions[currentQuestion].options.find(opt => opt.id === selectedOption).isCorrect;
        if (isCorrect) {
            setScore(score + 1);
        }

        if (currentQuestion + 1 < questions.length) {
            setCurrentQuestion(currentQuestion + 1);
            setSelectedOption(null);
        } else {
            // Finish Quiz
            const finalScore = isCorrect ? score + 1 : score;
            setShowResult(true);

            // Auto-complete lesson if passed (e.g. > 60%)
            if ((finalScore / questions.length) >= 0.6) {
                markLessonComplete(courseId, lesson.id);
            }
        }
    };

    const resetQuiz = () => {
        setCurrentQuestion(0);
        setSelectedOption(null);
        setScore(0);
        setShowResult(false);
        setQuizStarted(false);
    };

    if (!quizStarted) {
        return (
            <div className="flex flex-col items-center justify-center p-12 bg-white rounded-xl border border-slate-200 shadow-sm text-center">
                <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mb-6">
                    <AlertCircle className="w-8 h-8 text-primary-600" />
                </div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">{lesson.title}</h2>
                <p className="text-slate-600 mb-8 max-w-md">
                    Test your knowledge with this quick quiz. You need to score at least 60% to pass and complete this lesson.
                </p>
                <div className="flex gap-8 mb-8 text-sm font-semibold text-slate-500">
                    <div>{questions.length} Questions</div>
                    <div>5 Minutes</div>
                    <div>60% Passing Score</div>
                </div>
                <button
                    onClick={() => setQuizStarted(true)}
                    className="px-8 py-3 bg-primary-600 hover:bg-primary-700 text-white font-bold rounded-lg shadow-lg shadow-primary-500/30 transition-transform hover:scale-105"
                >
                    Start Quiz
                </button>
            </div>
        );
    }

    if (showResult) {
        const percentage = Math.round((score / questions.length) * 100);
        const passed = percentage >= 60;

        return (
            <div className="flex flex-col items-center justify-center p-12 bg-white rounded-xl border border-slate-200 shadow-sm text-center">
                <div className={`w-20 h-20 rounded-full flex items-center justify-center mb-6 ${passed ? 'bg-green-100' : 'bg-red-100'}`}>
                    {passed ? (
                        <CheckCircle className="w-10 h-10 text-green-600" />
                    ) : (
                        <XCircle className="w-10 h-10 text-red-600" />
                    )}
                </div>
                <h2 className="text-3xl font-bold text-slate-900 mb-2">
                    {passed ? 'Quiz Passed!' : 'Quiz Failed'}
                </h2>
                <div className="text-5xl font-bold text-slate-800 mb-4">{percentage}%</div>
                <p className="text-slate-600 mb-8 max-w-md">
                    {passed
                        ? "Great job! You have successfully demonstrated your understanding of this topic."
                        : "Don't worry, you can retake the quiz to improve your score. Review the lesson materials and try again."}
                </p>

                <div className="flex gap-4">
                    {!passed && (
                        <button
                            onClick={resetQuiz}
                            className="px-6 py-2 border border-slate-300 text-slate-700 font-bold rounded-lg hover:bg-slate-50 flex items-center gap-2"
                        >
                            <RefreshCw className="w-4 h-4" /> Retake Quiz
                        </button>
                    )}
                    {passed && (
                        <button className="px-6 py-2 bg-green-600 text-white font-bold rounded-lg shadow-sm cursor-default">
                            Lesson Completed
                        </button>
                    )}
                </div>
            </div>
        );
    }

    return (
        <div className="max-w-2xl mx-auto bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            {/* Progress Bar */}
            <div className="w-full bg-slate-100 h-2">
                <div
                    className="bg-primary-600 h-2 transition-all duration-300"
                    style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
                ></div>
            </div>

            <div className="p-8">
                <div className="flex justify-between items-center mb-6">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                        Question {currentQuestion + 1} of {questions.length}
                    </span>
                    <span className="text-xs font-semibold text-slate-400">
                        Score: {score}
                    </span>
                </div>

                <h3 className="text-xl font-bold text-slate-900 mb-8 leading-relaxed">
                    {questions[currentQuestion].questionText}
                </h3>

                <div className="space-y-3 mb-8">
                    {questions[currentQuestion].options.map(option => (
                        <button
                            key={option.id}
                            onClick={() => handleOptionSelect(option.id)}
                            className={`w-full text-left p-4 rounded-lg border-2 transition-all ${selectedOption === option.id
                                    ? 'border-primary-600 bg-primary-50 text-primary-900'
                                    : 'border-slate-100 hover:border-slate-300 text-slate-700'
                                }`}
                        >
                            <span className={`inline-block w-6 h-6 rounded-full border-2 mr-3 align-middle ${selectedOption === option.id ? 'border-primary-600 bg-primary-600' : 'border-slate-300'}`}></span>
                            {option.text}
                        </button>
                    ))}
                </div>

                <div className="flex justify-end">
                    <button
                        onClick={handleNextQuestion}
                        disabled={!selectedOption}
                        className={`px-6 py-3 rounded-lg font-bold flex items-center gap-2 transition-colors ${!selectedOption
                                ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                                : 'bg-primary-600 text-white hover:bg-primary-700 shadow-md'
                            }`}
                    >
                        {currentQuestion + 1 === questions.length ? 'Finish Quiz' : 'Next Question'}
                        <ChevronRight className="w-4 h-4" />
                    </button>
                </div>
            </div>
        </div>
    );
}
