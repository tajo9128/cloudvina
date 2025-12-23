import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import SEOHelmet from '../components/SEOHelmet'
import {
    Activity, GraduationCap, Factory, FlaskConical, Stethoscope,
    ArrowRight, CheckCircle2, Zap, Search, Globe, ShieldCheck,
    CheckCircle, Play, Server
} from 'lucide-react'

export default function HomePage() {
    const [isVisible, setIsVisible] = useState(false)

    useEffect(() => {
        setIsVisible(true)
    }, [])

    return (
        <div className="overflow-hidden bg-slate-50 font-sans text-slate-900">
            <SEOHelmet
                title="BioDockify | Intelligent Molecular Research Platform"
                description="Accelerating Drug Discovery Through Intelligent Molecular Research. Simplify and accelerate modern drug discovery research without expensive infrastructure."
                keywords="drug discovery, molecular analysis, computational pharmacology, virtual screening, medicinal chemistry"
                canonical="https://biodockify.com/"
            />

            {/* 1. CLOUD-NATIVE HERO SECTION (Restored) */}
            <section className="relative bg-slate-900 overflow-hidden pt-20 pb-32 lg:pt-32 lg:pb-40">
                {/* Background Glows */}
                <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-indigo-600/20 rounded-full blur-[120px] -translate-y-1/2 translate-x-1/3"></div>
                <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-emerald-600/10 rounded-full blur-[120px] translate-y-1/2 -translate-x-1/3"></div>

                <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="lg:grid lg:grid-cols-2 gap-16 items-center">

                        {/* LEFT: Text Content */}
                        <div className="max-w-2xl mb-12 lg:mb-0">
                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-900/50 border border-indigo-700/50 text-indigo-300 text-[10px] font-bold uppercase tracking-wider mb-8 animate-fade-in">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                                Major Release v6.0.0: Privacy-First Consensus Docking
                            </div>

                            <h1 className="text-5xl lg:text-7xl font-bold text-white mb-6 leading-[1.1] tracking-tight">
                                Cloud-Native, <br />
                                <span className="text-indigo-400">End-to-End</span> <br />
                                Drug Discovery.
                            </h1>

                            <p className="text-lg text-slate-400 mb-10 leading-relaxed max-w-lg">
                                Accelerate your pipeline from Virtual Screening to Lead Optimization with our high-performance cloud infrastructure.
                                Scalable, secure, and ready for production.
                            </p>

                            <div className="flex flex-wrap gap-4">
                                <Link to="/dock/new" className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white font-bold rounded-xl shadow-lg shadow-indigo-900/20 transition-all hover:-translate-y-0.5">
                                    <Play size={20} fill="currentColor" />
                                    <span>Start Free Trial</span>
                                </Link>
                                <Link to="/contact" className="inline-flex items-center justify-center gap-2 px-8 py-4 bg-slate-800/50 hover:bg-slate-800 text-white font-semibold rounded-xl border border-slate-700 transition-all group">
                                    <span>Contact Sales</span>
                                    <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
                                </Link>
                            </div>
                        </div>

                        {/* RIGHT: Visual */}
                        <div className="relative animate-fade-in-up delay-200">
                            <div className="relative rounded-3xl overflow-hidden border border-slate-700/50 shadow-2xl bg-slate-800/50 backdrop-blur-sm group">
                                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-1000"></div>

                                {/* Floating Badge */}
                                <div className="absolute top-8 left-8 z-20 bg-slate-900/90 backdrop-blur-md p-4 rounded-xl border border-slate-700 shadow-xl flex items-center gap-4 max-w-[240px]">
                                    <div className="p-2 bg-emerald-500/20 rounded-lg">
                                        <Server size={24} className="text-emerald-400" />
                                    </div>
                                    <div>
                                        <div className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Secure Storage</div>
                                        <div className="text-sm font-bold text-white">Encrypted & Compliant</div>
                                    </div>
                                </div>

                                <img
                                    src="/assets/images/hero-molecular-v36.png"
                                    alt="BioDockify Interface"
                                    className="w-full h-auto transform transition-transform duration-700 hover:scale-105"
                                    onError={(e) => {
                                        console.error("Hero image failed to load, falling back to placeholder");
                                        e.target.onerror = null;
                                        e.target.src = "https://images.unsplash.com/photo-1614935151651-0bea6508db6b?auto=format&fit=crop&q=80&w=1200";
                                    }}
                                />

                                {/* Overlay Text on Image (Bottom Right) */}
                                <div className="absolute bottom-0 right-0 p-8 text-right bg-gradient-to-t from-slate-900 via-slate-900/50 to-transparent w-full">
                                    <div className="text-2xl font-bold text-white tracking-widest uppercase opacity-80">Biotech Innovations</div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </section>

            {/* 2. THE 3 PILLARS OF DISCOVERY */}
            <section id="workflow" className="py-32 bg-slate-50 relative overflow-hidden">
                {/* Background Decor */}
                <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
                    <div className="absolute top-1/4 left-0 w-[800px] h-[800px] bg-indigo-200/20 rounded-full blur-[120px] -translate-x-1/2"></div>
                    <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-emerald-200/20 rounded-full blur-[100px] translate-x-1/3"></div>
                </div>

                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="text-center mb-20">
                        <h2 className="text-sm font-bold text-indigo-600 tracking-[0.2em] uppercase mb-4">Complete 9-Phase Pipeline</h2>
                        <h3 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6">Three Pillars of Discovery</h3>
                        <p className="text-lg text-slate-600 max-w-4xl mx-auto leading-relaxed">
                            We have consolidated the entire discovery lifecycle into three integrated workspaces:
                            <span className="block mt-4 text-slate-500 font-medium border-t border-b border-slate-200 py-4 bg-white/50">
                                1. Target ID &nbsp;•&nbsp; 2. Docking &nbsp;•&nbsp; 3. MD Simulation &nbsp;•&nbsp; 4. Trajectory Analysis &nbsp;•&nbsp; 5. Binding Energy <br className="hidden md:block" />
                                6. Lead Ranking &nbsp;•&nbsp; 7. ADMET Profiling &nbsp;•&nbsp; 8. Benchmarking &nbsp;•&nbsp; 9. Reporting
                            </span>
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 gap-6">
                        {/* PILLAR 1: AI VIRTUAL SCREENING (CNS Ensemble B) */}
                        <PillarCard
                            number="01"
                            title="AI Virtual Screening"
                            subtitle="Fine Tuned AI Ensemble"
                            description="Multi-encoder deep learning for CNS targets. Fused ChemBERTa (SMILES) + GNN (Structure) + ProteinCNN architecture trained on Google Colab L4 GPU for affinity prediction."
                            features={[
                                "ChemBERTa 768-dim Molecular Embeddings",
                                "GNN (GAT) Structure-Aware Features",
                                "DeepDTA pKd Affinity Prediction",
                                "Blood-Brain Barrier (BBB) Classification",
                                "Plant Alkaloid Phytochemical Library"
                            ]}
                            icon={<Zap className="w-8 h-8 text-white" />}
                            color="indigo"
                            link="https://ai.biodockify.com"
                            linkText="Launch Screening"
                        />

                        {/* PILLAR 2: HIGH-THROUGHPUT DOCKING */}
                        <PillarCard
                            number="02"
                            title="High-Throughput Docking"
                            subtitle="Consensus Dual-Engine"
                            description="Massive-scale validation with parallel AutoDock Vina and Gnina execution. Secure batch processing with real-time protein preparation and automated grid calculation."
                            features={[
                                "Consensus Scoring (Vina + Gnina)",
                                "Automated Protein Prep Pipeline",
                                "Parallel Cloud Execution (5 cores)",
                                "PDBQT Conversion & Grid Generation",
                                "Real-Time Terminal & Progress Logs"
                            ]}
                            icon={<Server className="w-8 h-8 text-white" />}
                            color="blue"
                            link="/dock/new"
                            linkText="Start Docking"
                        />

                        {/* PILLAR 3: DYNAMICS ENGINE */}
                        <PillarCard
                            number="03"
                            title="Molecular Dynamics"
                            subtitle="Phases 3-5: Physics Engine"
                            description="Full-atomistic MD simulations with Amber14/OpenMM force fields. Calculate binding free energies via MM-PBSA and validate protein-ligand stability under explicit solvation."
                            features={[
                                "Amber14 Force Fields (ff14SB)",
                                "Explicit Solvent MD (TIP3P Water)",
                                "Production Runs (10-100 ns)",
                                "Trajectory RMSD/RMSF Analysis",
                                "MM-PBSA Free Energy Calculation"
                            ]}
                            icon={<Activity className="w-8 h-8 text-white" />}
                            color="amber"
                            link="/dock/new"
                            linkText="Run Simulation"
                        />

                        {/* PILLAR 4: LEAD DISCOVERY */}
                        <PillarCard
                            number="04"
                            title="Lead Discovery Workspace"
                            subtitle="Phases 6-7: Multi-Parameter"
                            description="Consensus-based lead ranking integrating docking scores, MD stability, and AI predictions. Full ADMET toxicity profiling with Lipinski rule validation."
                            features={[
                                "Consensus Ranking (Docking + MD + AI)",
                                "ADMET: hERG, Ames, CYP450 Inhibition",
                                "Lipinski Rule of 5 Compliance",
                                "BBB Penetration Prediction",
                                "Synthetic Accessibility Score (SAScore)"
                            ]}
                            icon={<Search className="w-8 h-8 text-white" />}
                            color="violet"
                            link="/leads"
                            linkText="Explore Workspace"
                        />

                        {/* PILLAR 5: PLATFORM ASSURANCE */}
                        <PillarCard
                            number="05"
                            title="Platform Assurance"
                            subtitle="Phases 8-9: Regulatory QA"
                            description="Continuous model benchmarking against experimental truth. Generate FDA 21 CFR Part 11 compliant audit logs and automated validation reports."
                            features={[
                                "R² & RMSE Accuracy Benchmarking",
                                "Experimental vs Predicted Correlation",
                                "PDBbind / BindingDB Validation Sets",
                                "FDA 21 CFR Part 11 Audit Trails",
                                "Automated PDF Report Generation"
                            ]}
                            icon={<ShieldCheck className="w-8 h-8 text-white" />}
                            color="emerald"
                            link="/tools/benchmarking"
                            linkText="Run Validation"
                        />

                        {/* PILLAR 6: AI FORMULATION (BioDockify-Formulate) */}
                        <PillarCard
                            number="06"
                            title="AI Formulation Engine"
                            subtitle="Formulate-AI: 7-Model Stack"
                            description="Federated AI system predicting ANDA readiness. ChemBERTa API embeddings, XGBoost pre-formulation risk, GNN excipient compatibility, LSTM dissolution kinetics, and survival ML stability prediction."
                            features={[
                                "ChemBERTa API Property Prediction",
                                "XGBoost BCS Class & Polymorphism Risk",
                                "GNN Excipient Compatibility Ranking",
                                "LSTM Dissolution Profile Prediction",
                                "DeepSurv Shelf-Life Stability"
                            ]}
                            icon={<FlaskConical className="w-8 h-8 text-white" />}
                            color="cyan"
                            link="/formulation"
                            linkText="Design Formulation"
                        />
                    </div>
                </div>
            </section>

            {/* 3. WHY BIODOCKIFY (Challenges & Advantages) */}
            <section className="py-24 bg-white border-t border-slate-100">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="lg:grid lg:grid-cols-2 gap-16 items-center">
                        <div className="mb-12 lg:mb-0">
                            <h2 className="text-sm font-bold text-indigo-600 tracking-widest uppercase mb-3">Why BioDockify?</h2>
                            <h3 className="text-3xl md:text-5xl font-bold text-slate-900 mb-6">Bridging the Gap Between Research and Discovery</h3>
                            <p className="text-lg text-slate-600 mb-8 leading-relaxed">
                                Drug discovery today faces major challenges—high costs, long timelines, fragmented tools, and limited accessibility.
                                BioDockify addresses these by providing a unified environment where scientists can focus on discovery rather than software complexity.
                            </p>
                            <div className="space-y-6">
                                <div className="flex items-start gap-4">
                                    <div className="p-2 bg-emerald-100 rounded-lg text-emerald-600 mt-1"><CheckCircle2 size={20} /></div>
                                    <div>
                                        <h4 className="font-bold text-lg text-slate-900">Faster Molecular Analysis</h4>
                                        <p className="text-slate-500">Rapid compound evaluation and screening.</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="p-2 bg-indigo-100 rounded-lg text-indigo-600 mt-1"><Globe size={20} /></div>
                                    <div>
                                        <h4 className="font-bold text-lg text-slate-900">Accessible Anywhere</h4>
                                        <p className="text-slate-500">Cloud-native with no hardware dependency.</p>
                                    </div>
                                </div>
                                <div className="flex items-start gap-4">
                                    <div className="p-2 bg-amber-100 rounded-lg text-amber-600 mt-1"><Zap size={20} /></div>
                                    <div>
                                        <h4 className="font-bold text-lg text-slate-900">For All Skill Levels</h4>
                                        <p className="text-slate-500">Designed for both beginners and advanced researchers.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {/* 3D Visual for Section 3 */}
                        <div className="relative">
                            <div className="absolute inset-0 bg-blue-600 blur-[80px] opacity-10 rounded-full"></div>
                            <img
                                src="/assets/images/bridging-gap-v36.png"
                                alt="BioDockify Bridging Research and Discovery"
                                className="relative w-full h-auto rounded-3xl shadow-2xl border border-slate-200 transform hover:scale-[1.02] transition-transform duration-500"
                                onError={(e) => {
                                    e.target.onerror = null;
                                    e.target.src = "https://placehold.co/800x600/f8fafc/e2e8f0?text=Analysis+Interface";
                                }}
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* 4. WHO CAN USE (Personas) */}
            <section className="py-24 bg-slate-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">Who Can Use BioDockify?</h2>
                        <p className="text-lg text-slate-600">Start-to-finish support for the entire scientific community.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <PersonaCard
                            icon={<GraduationCap className="w-8 h-8 text-indigo-600" />}
                            title="PhD Scholars & Academia"
                            desc="Conduct computational studies, validate hypotheses, and support publications with reliable molecular insights."
                            color="indigo"
                        />
                        <PersonaCard
                            icon={<Factory className="w-8 h-8 text-emerald-600" />}
                            title="Pharma & Biotech"
                            desc="Accelerate lead identification and reduce early-stage R&D timelines with scalable screening."
                            color="emerald"
                        />
                        <PersonaCard
                            icon={<FlaskConical className="w-8 h-8 text-violet-600" />}
                            title="CROs & Research Labs"
                            desc="Perform high-throughput screening and molecular analysis for multiple clients efficiently."
                            color="violet"
                        />
                        <PersonaCard
                            icon={<Stethoscope className="w-8 h-8 text-blue-600" />}
                            title="Medicinal Chemistry"
                            desc="Explore compound behavior, binding potential, and structure-activity relationships."
                            color="blue"
                        />
                    </div>
                </div>
            </section>

            {/* 5. FINAL CTA */}
            <section className="py-32 bg-[#0B1121] text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-10"></div>
                <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] bg-indigo-600/20 rounded-full blur-[120px]"></div>
                <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-emerald-600/20 rounded-full blur-[120px]"></div>

                <div className="max-w-4xl mx-auto px-4 relative z-10">
                    <h2 className="text-5xl font-bold text-white mb-8">Start Your Research with Confidence</h2>
                    <p className="text-xl text-slate-300 mb-12 max-w-2xl mx-auto leading-relaxed">
                        Reduce experimental risk, improve decision-making, and accelerate discovery timelines.
                        From academic exploration to industry-ready insights.
                    </p>
                    <div className="flex flex-col sm:flex-row justify-center gap-6">
                        <Link to="/dock/new" className="px-10 py-5 bg-white text-slate-900 font-bold text-lg rounded-2xl shadow-xl hover:bg-indigo-50 transition-all transform hover:-translate-y-1">
                            Begin Your Research
                        </Link>
                        <Link to="/contact" className="px-10 py-5 bg-indigo-900/50 text-white font-bold text-lg rounded-2xl border border-indigo-700/50 hover:bg-indigo-800 transition-all backdrop-blur-sm">
                            Contact Us
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    )
}

// --- Subcomponents ---

function PillarCard({ number, title, subtitle, description, features, icon, color, link, linkText }) {
    const colorStyles = {
        indigo: {
            bg: 'bg-indigo-50',
            hoverBorder: 'hover:border-indigo-200',
            iconBg: 'bg-indigo-600',
            text: 'text-indigo-600',
            btn: 'bg-indigo-600 hover:bg-indigo-700'
        },
        violet: {
            bg: 'bg-violet-50',
            hoverBorder: 'hover:border-violet-200',
            iconBg: 'bg-violet-600',
            text: 'text-violet-600',
            btn: 'bg-violet-600 hover:bg-violet-700'
        },
        emerald: {
            bg: 'bg-emerald-50',
            hoverBorder: 'hover:border-emerald-200',
            iconBg: 'bg-emerald-600',
            text: 'text-emerald-600',
            btn: 'bg-emerald-600 hover:bg-emerald-700'
        },
        blue: {
            bg: 'bg-blue-50',
            hoverBorder: 'hover:border-blue-200',
            iconBg: 'bg-blue-600',
            text: 'text-blue-600',
            btn: 'bg-blue-600 hover:bg-blue-700'
        },
        amber: {
            bg: 'bg-amber-50',
            hoverBorder: 'hover:border-amber-200',
            iconBg: 'bg-amber-600',
            text: 'text-amber-600',
            btn: 'bg-amber-600 hover:bg-amber-700'
        },
        cyan: {
            bg: 'bg-cyan-50',
            hoverBorder: 'hover:border-cyan-200',
            iconBg: 'bg-cyan-600',
            text: 'text-cyan-600',
            btn: 'bg-cyan-600 hover:bg-cyan-700'
        }
    };

    const style = colorStyles[color];
    const isExternal = link.startsWith('http');

    const ButtonContent = () => (
        <>
            <span className={`font-bold ${style.text}`}>{linkText}</span>
            <ArrowRight size={18} className={`${style.text} transform group-hover/btn:translate-x-1 transition-transform`} />
        </>
    );

    const buttonClass = `flex items-center justify-between w-full px-5 py-4 rounded-xl ${style.bg} hover:bg-white border border-transparent hover:border-slate-200 group/btn transition-all`;

    return (
        <div className={`relative bg-white rounded-3xl p-8 shadow-sm border border-slate-100 transition-all duration-300 hover:shadow-2xl hover:-translate-y-2 ${style.hoverBorder} group overflow-hidden flex flex-col`}>
            {/* Top Decor */}
            <div className={`absolute top-0 right-0 w-32 h-32 ${style.bg} rounded-bl-full -mr-8 -mt-8 opacity-50 group-hover:scale-110 transition-transform`}></div>

            <div className="relative z-10 flex-1">
                <div className="flex justify-between items-start mb-6">
                    <div className={`p-3 rounded-xl ${style.iconBg} shadow-lg shadow-${color}-900/20`}>
                        {icon}
                    </div>
                    <span className="text-6xl font-black text-slate-100 select-none group-hover:text-slate-200 transition-colors">{number}</span>
                </div>

                <div className="mb-6">
                    <h3 className="text-2xl font-bold text-slate-900 mb-1">{title}</h3>
                    <div className={`text-xs font-bold ${style.text} uppercase tracking-widest`}>{subtitle}</div>
                </div>

                <p className="text-slate-600 mb-8 leading-relaxed">
                    {description}
                </p>

                <div className="space-y-3 mb-8">
                    {features.map((feature, i) => (
                        <div key={i} className="flex items-center gap-3 text-sm text-slate-600">
                            <CheckCircle2 size={16} className={`${style.text} shrink-0`} />
                            <span>{feature}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div className="relative z-10 mt-auto">
                {isExternal ? (
                    <a href={link} className={buttonClass} target="_blank" rel="noopener noreferrer">
                        <ButtonContent />
                    </a>
                ) : (
                    <Link to={link} className={buttonClass}>
                        <ButtonContent />
                    </Link>
                )}
            </div>
        </div>
    )
}

function PersonaCard({ icon, title, desc, color }) {
    const bgColors = {
        indigo: 'bg-indigo-50 hover:bg-indigo-100',
        emerald: 'bg-emerald-50 hover:bg-emerald-100',
        violet: 'bg-violet-50 hover:bg-violet-100',
        blue: 'bg-blue-50 hover:bg-blue-100'
    }

    return (
        <div className={`p-8 rounded-2xl ${bgColors[color]} transition-colors text-center border border-slate-100`}>
            <div className="w-16 h-16 mx-auto bg-white rounded-full flex items-center justify-center shadow-sm mb-6">
                {icon}
            </div>
            <h3 className="text-lg font-bold text-slate-900 mb-3">{title}</h3>
            <p className="text-slate-600 text-sm leading-relaxed">{desc}</p>
        </div>
    )
}
