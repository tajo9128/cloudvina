import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Box, Zap, Shield, MousePointer, Layers, Cpu, Database, Activity } from 'lucide-react'
import Footer from '../components/Footer'

export default function MolecularDockingPage() {
    return (
        <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
            {/* HER HERO SECTION */}
            <div className="relative bg-[#0B1121] text-white overflow-hidden">
                {/* Background Gradients */}
                <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0">
                    <div className="absolute top-[-10%] right-[-5%] w-[50%] h-[50%] bg-primary-600/20 rounded-full blur-[120px] animate-pulse-slow"></div>
                    <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-secondary-600/20 rounded-full blur-[100px] animate-pulse-slow delay-1000"></div>
                </div>

                <div className="container mx-auto px-4 pt-32 pb-24 relative z-10">
                    <div className="max-w-4xl mx-auto text-center">
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800/50 border border-slate-700 backdrop-blur-sm mb-6">
                            <span className="flex h-2 w-2 rounded-full bg-green-400 animate-pulse"></span>
                            <span className="text-sm font-medium text-slate-300">Powered by AutoDock Vina</span>
                        </div>
                        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8 leading-tight">
                            Advanced <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-secondary-400">Molecular Docking</span> Online
                        </h1>
                        <p className="text-xl text-slate-300 mb-10 max-w-2xl mx-auto leading-relaxed">
                            Accelerate your drug discovery research with our cloud-native docking platform.
                            Perform high-accuracy interactions analysis without expensive hardware.
                        </p>
                        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                            <Link to="/dock/batch" className="px-8 py-4 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-bold text-lg shadow-lg shadow-primary-600/25 transition-all hover:-translate-y-1 flex items-center gap-2">
                                Start Docking Now <ArrowRight size={20} />
                            </Link>
                            <Link to="/pricing" className="px-8 py-4 bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 rounded-xl font-bold text-lg transition-all hover:-translate-y-1">
                                View Pricing
                            </Link>
                        </div>
                    </div>
                </div>
            </div>

            {/* MAIN CONTENT CONTAINER */}
            <div className="container mx-auto px-4 py-16 max-w-7xl">

                {/* KEY BENEFITS GRID */}
                <div className="grid md:grid-cols-3 gap-8 mb-24 -mt-24 relative z-20">
                    {[
                        {
                            icon: <Zap className="w-8 h-8 text-yellow-400" />,
                            title: "Auto Vina Online Engine",
                            desc: "Leverage the industry-standard AutoDock Vina algorithm optimized for cloud execution. Get results in minutes, not hours."
                        },
                        {
                            icon: <Database className="w-8 h-8 text-blue-400" />,
                            title: "High-Throughput Ready",
                            desc: "Screen thousands of compounds simultaneously. Our scalable infrastructure handles batch processing with ease."
                        },
                        {
                            icon: <Box className="w-8 h-8 text-green-400" />,
                            title: "3D Visualization",
                            desc: "Instantly visualize protein-ligand interactions in your browser with our integrated NGL-based viewer."
                        }
                    ].map((feature, idx) => (
                        <div key={idx} className="bg-[#1e293b] p-8 rounded-2xl border border-slate-700 shadow-xl hover:border-primary-500/50 transition-colors group">
                            <div className="bg-slate-800/50 w-16 h-16 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                                {feature.icon}
                            </div>
                            <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
                            <p className="text-slate-400 leading-relaxed">{feature.desc}</p>
                        </div>
                    ))}
                </div>

                {/* SEO CONTENT SECTION 1 */}
                <div className="grid lg:grid-cols-2 gap-16 items-center mb-24">
                    <div>
                        <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-6">
                            What is Molecular Docking Online?
                        </h2>
                        <div className="prose prose-lg text-slate-600">
                            <p className="mb-4">
                                <strong>Molecular docking online</strong> serves as a pivotal computational technique in modern structural biology and computer-aided drug design (CADD). It predicts the preferred orientation of a ligand (typically a small molecule drug candidate) to a receptor (usually a protein) when bound to each other to form a stable complex.
                            </p>
                            <p className="mb-4">
                                Traditional docking requires high-performance local workstations and complex software configurations. BioDockify revolutionizes this by providing a seamless <strong>Auto Vina online</strong> experience. Our platform abstracts the complexity of command-line tools, offering a user-friendly interface that runs powerful algorithms on improved cloud infrastructure.
                            </p>
                            <p>
                                By calculating the binding free energy, researchers can determine the strength of the association (affinity) between the ligand and protein. This predictive capability significantly reduces the time and cost associated with experimental lab screening, making it an essential tool for pharmaceutical researchers, biochemists, and academic students alike.
                            </p>
                        </div>
                    </div>
                    <div className="bg-slate-100 rounded-2xl p-2 border border-slate-200 shadow-inner">
                        <img
                            src="https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&q=80&w=1000"
                            alt="Molecular Docking Simulation Interface"
                            className="rounded-xl shadow-lg w-full h-auto object-cover"
                        />
                        <p className="text-center text-xs text-slate-400 mt-2">High-accuracy protein-ligand binding simulation</p>
                    </div>
                </div>

                {/* FEATURES LIST */}
                <div className="mb-24">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">Why Choose BioDockify?</h2>
                        <p className="text-xl text-slate-500 max-w-2xl mx-auto">
                            We combine scientific rigor with modern software engineering to deliver the best docking experience.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {[
                            { title: "No Installation", desc: "Run directly in your browser. No Linux terminal required." },
                            { title: "Binding Affinity", desc: "Accurate kcal/mol scores calculated for every pose." },
                            { title: "Secure Data", desc: "Your proprietary molecular data is encrypted and private." },
                            { title: "Export Ready", desc: "Download results in PDBQT, CSV, and high-res images." },
                            { title: "Blind Docking", desc: "Support for whole-protein search box definitions." },
                            { title: "Flexible Grid", desc: "Interactive 3D box selection for targeted docking." },
                            { title: "Multi-Ligand", desc: "Upload multiple ligands and screen them in one batch." },
                            { title: "History Saved", desc: "Access your past jobs and results anytime from the dashboard." }
                        ].map((item, i) => (
                            <div key={i} className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                                <div className="flex items-center gap-3 mb-3">
                                    <div className="h-2 w-2 rounded-full bg-primary-500"></div>
                                    <h4 className="font-bold text-slate-800">{item.title}</h4>
                                </div>
                                <p className="text-sm text-slate-500">{item.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* SEO CONTENT SECTION 2 */}
                <div className="bg-slate-900 rounded-3xl p-8 md:p-12 text-white overflow-hidden relative mb-24">
                    {/* Background Pattern */}
                    <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-primary-600/10 rounded-full blur-[100px] pointer-events-none"></div>

                    <h2 className="text-3xl font-bold mb-8 relative z-10">The Science Behind AutoDock Vina</h2>
                    <div className="grid md:grid-cols-2 gap-12 relative z-10">
                        <div className="prose prose-invert prose-lg">
                            <p>
                                <strong>AutoDock Vina</strong> is one of the most cited and trusted docking engines in the world. It uses a sophisticated scoring function and an efficient optimization algorithm to explore the conformational space of the ligand within the protein's binding site.
                            </p>
                            <h4 className="text-white font-bold mt-6 mb-2">Empirical Scoring Function</h4>
                            <p className="text-slate-300 text-base">
                                Vina's scoring function approximates the standard chemical potential of the system. It accounts for steric interactions (Gauss 1, Gauss 2), repulsion, hydrogen bonding, and hydrophobic interactions. This ensures that the predicted poses are energetically favorable and physically realistic.
                            </p>
                        </div>
                        <div className="prose prose-invert prose-lg">
                            <h4 className="text-white font-bold mb-2">Why Online is Better?</h4>
                            <p className="text-slate-300 text-base mb-6">
                                Running <strong>Auto vina online</strong> eliminates the computational bottleneck of local machines. Complex docking jobs that might freeze a standard laptop run smoothly on our optimized cloud clusters. BioDockify handles the PDBQT preparation, grid box generation, and error handling automatically.
                            </p>
                            <ul className="space-y-2 text-slate-300">
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-1">✓</span>
                                    <span>Faster convergence with iterated local search global optimizer.</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-1">✓</span>
                                    <span>Parallelized execution for batch operations.</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-1">✓</span>
                                    <span>Regular updates with the latest force field parameters.</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* STEPS CTA */}
                <div className="text-center max-w-3xl mx-auto mb-24">
                    <h2 className="text-3xl font-bold text-slate-900 mb-12">How It Works</h2>
                    <div className="flex flex-col md:flex-row justify-between items-center gap-8 relative">
                        {/* Connecting Line */}
                        <div className="hidden md:block absolute top-1/2 left-0 w-full h-1 bg-slate-200 -z-10 -translate-y-1/2"></div>

                        {[
                            { num: "01", title: "Upload PDB", icon: <Layers size={24} /> },
                            { num: "02", title: "Select Ligand", icon: <Activity size={24} /> },
                            { num: "03", title: "Run Vina", icon: <Cpu size={24} /> },
                            { num: "04", title: "Analyze", icon: <MousePointer size={24} /> },
                        ].map((step, idx) => (
                            <div key={idx} className="bg-white p-6 rounded-2xl border border-slate-200 shadow-lg w-full md:w-48 relative">
                                <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-primary-600 text-white text-xs font-bold py-1 px-3 rounded-full">
                                    STEP {step.num}
                                </div>
                                <div className="text-primary-600 mb-3 flex justify-center">{step.icon}</div>
                                <h4 className="font-bold text-slate-900">{step.title}</h4>
                            </div>
                        ))}
                    </div>
                </div>

                {/* FAQ SECTION */}
                <div className="max-w-3xl mx-auto mb-24">
                    <h2 className="text-3xl font-bold text-slate-900 mb-8 text-center">Frequently Asked Questions</h2>
                    <div className="space-y-4">
                        {[
                            {
                                q: "Is this molecular docking tool free?",
                                a: "We offer a Free Tier that includes 3 docking jobs per day suitable for students and hobbyists. For advanced research and unlimited batch processing, check out our implementation of Pro and Enterprise plans."
                            },
                            {
                                q: "Do I need to install AutoDock Vina?",
                                a: "No! BioDockify runs AutoDock Vina online in the cloud. You only need a web browser and your internet connection. We handle all the software dependencies and updates."
                            },
                            {
                                q: "What file formats are supported?",
                                a: "We support standard PDB files for receptor proteins and PDB/SDF/MOL2 files for ligands. Our system automatically converts them to the required PDBQT format for docking."
                            },
                            {
                                q: "Can I use this for academic papers?",
                                a: "Absolutely. BioDockify uses the standard, validated AutoDock Vina engine. You can cite AutoDock Vina and mention BioDockify as the computing platform in your methodology section."
                            }
                        ].map((faq, i) => (
                            <div key={i} className="bg-white p-6 rounded-xl border border-slate-200">
                                <h3 className="font-bold text-slate-900 mb-2">{faq.q}</h3>
                                <p className="text-slate-600">{faq.a}</p>
                            </div>
                        ))}
                    </div>
                </div>


            </div>

            <Footer />
        </div>
    )
}
