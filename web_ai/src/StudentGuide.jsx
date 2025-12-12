import React from 'react';
import { BookOpen, MonitorPlay, Zap, Brain, Activity, ArrowLeft } from 'lucide-react';

export default function StudentGuide({ onBack }) {
    return (
        <div className="min-h-screen bg-black text-slate-100 font-sans p-6 md:p-12">
            <div className="max-w-4xl mx-auto">
                <button
                    onClick={onBack}
                    className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors mb-8 group"
                >
                    <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" /> Back to Workbench
                </button>

                <div className="bg-slate-900/50 border border-white/10 rounded-2xl p-8 md:p-12 shadow-2xl backdrop-blur-sm">
                    {/* Header */}
                    <div className="flex items-center gap-4 mb-8 border-b border-white/10 pb-8">
                        <div className="w-16 h-16 bg-indigo-600/20 rounded-2xl flex items-center justify-center border border-indigo-500/30">
                            <BookOpen className="w-8 h-8 text-indigo-400" />
                        </div>
                        <div>
                            <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">Student Introduction Guide</h1>
                            <p className="text-slate-400 text-lg">"Drug Discovery used to cost Millions. Now it's Free on your Laptop."</p>
                        </div>
                    </div>

                    {/* Section 1: The Hook */}
                    <section className="mb-12">
                        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                            <Zap className="text-amber-400" /> The Hook: Why Should They Care?
                        </h2>
                        <div className="bg-black/30 p-6 rounded-xl border border-white/5">
                            <p className="text-slate-300 leading-relaxed mb-4">
                                Most students think Drug Discovery requires a multi-million dollar lab with clean rooms and robotic arms.
                                <strong className="text-indigo-400"> BioDockify AI proves them wrong.</strong>
                            </p>
                            <p className="text-slate-400">
                                This platform democratizes "Big Pharma" tools—Artificial Intelligence, Cloud Computing, and Generative Design—and puts them directly in their browser.
                                It turns a laptop into a molecular research facility.
                            </p>
                        </div>
                    </section>

                    {/* Section 2: Workshop Flow */}
                    <section className="mb-12">
                        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                            <MonitorPlay className="text-emerald-400" /> The 1-Hour Workshop Flow
                        </h2>

                        <div className="space-y-4">
                            <WorkshopStep
                                num="01"
                                title="The Digitization (Cheminformatics)"
                                desc="Have students draw a molecule like Aspirin, find its SMILES string, and upload it. Lesson: 'Molecules are just text to a computer.'"
                                icon={<BookOpen size={18} />}
                            />
                            <WorkshopStep
                                num="02"
                                title="The Safety Check (Toxicity AI)"
                                desc="Feed the AI a known poison (Benzene) and a medicine. Lesson: 'AI predicts safety outcomes instantly, reducing the need for animal testing.'"
                                icon={<Activity size={18} />}
                            />
                            <WorkshopStep
                                num="03"
                                title="The Invention (Generative AI)"
                                desc="Launch the 'Colab GPU Worker' to dream up new molecules. Lesson: 'AI can be creative, learning chemical grammar to write new molecular sentences.'"
                                icon={<Brain size={18} />}
                            />
                            <WorkshopStep
                                num="04"
                                title="The Loop (Active Learning)"
                                desc="Correct the AI's prediction and click Retrain. Lesson: 'AI is a partner we teach, not just a static tool.'"
                                icon={<Zap size={18} />}
                            />
                        </div>
                    </section>

                    {/* Section 3: Tech Stack */}
                    <section>
                        <h2 className="text-xl font-bold text-white mb-4">Under The Hood</h2>
                        <p className="text-slate-400 mb-6">
                            Explain to students that they are using professional open-source standards, not just a toy.
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <TechCard title="Hugging Face" desc="The 'GitHub of AI' hosting our neural networks." />
                            <TechCard title="PyTorch" desc="The same library used to build ChatGPT." />
                            <TechCard title="RDKit" desc=" The industry standard for chemical data." />
                        </div>
                    </section>
                </div>
            </div>
        </div>
    );
}

function WorkshopStep({ num, title, desc, icon }) {
    return (
        <div className="flex gap-4 p-4 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
            <div className="text-2xl font-bold text-white/10 font-mono">{num}</div>
            <div>
                <h3 className="text-lg font-semibold text-white mb-1 flex items-center gap-2">
                    {title}
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed">{desc}</p>
            </div>
        </div>
    )
}

function TechCard({ title, desc }) {
    return (
        <div className="p-4 rounded-lg bg-indigo-900/10 border border-indigo-500/20 text-center">
            <div className="font-bold text-indigo-400 mb-1">{title}</div>
            <div className="text-xs text-slate-500">{desc}</div>
        </div>
    )
}
