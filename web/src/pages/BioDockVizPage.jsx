
import React from 'react';
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import {
    Download,
    Wifi,
    ShieldCheck,
    Cpu,
    Activity,
    Layers,
    Terminal,
    CheckCircle,
    Monitor,
    Zap,
    Globe,
    FileText,
    MousePointer,
    Maximize
} from 'lucide-react';

const BioDockVizPage = () => {
    const [downloadUrl, setDownloadUrl] = React.useState('https://github.com/tajo9128/BioDockViz/releases/latest');
    const [version, setVersion] = React.useState('v1.0.0');

    // SEO setup & Fetch Download Link
    React.useEffect(() => {
        document.title = "BioDockViz - AI-Powered Molecular Docking Visualization & Analysis";
        const metaDesc = document.querySelector('meta[name="description"]');
        if (metaDesc) {
            metaDesc.setAttribute('content', "BioDockViz: A professional, web-based molecular visualization platform specific for docking results. AI-assisted analysis and publication-quality exports.");
        }

        // Fetch latest release
        const fetchLatestRelease = async () => {
            try {
                // Assuming a repo exists for BioDockViz, otherwise this will fail silently and use default
                const response = await fetch('https://api.github.com/repos/tajo9128/BioDockViz/releases/latest');
                if (response.ok) {
                    const data = await response.json();
                    setVersion(data.tag_name);
                    // Find .exe asset
                    const exeAsset = data.assets.find(asset => asset.name.endsWith('.exe'));
                    if (exeAsset) {
                        setDownloadUrl(exeAsset.browser_download_url);
                    }
                }
            } catch (error) {
                console.warn("Failed to fetch latest BioDockViz release (Repo might not exist yet):", error);
            }
        };

        fetchLatestRelease();
    }, []);

    return (
        <div className="bg-white text-gray-900 font-sans">
            {/* Hero Section */}
            <section className="relative overflow-hidden bg-emerald-900 text-white pt-20 pb-32">
                <div className="absolute inset-0 bg-gradient-to-br from-teal-900 to-emerald-900 opacity-90"></div>
                {/* Abstract Molecule Background Pattern */}
                <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)', backgroundSize: '40px 40px' }}></div>

                <div className="container mx-auto px-6 relative z-10 text-center">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-400/30 text-emerald-300 text-sm font-medium mb-6">
                        <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse"></span>
                        AI-Powered Visualization
                    </div>
                    <h1 className="text-4xl md:text-6xl font-bold mb-6 tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-emerald-200">
                        AI-Powered Molecular Docking<br />Visualization & Analysis
                    </h1>
                    <p className="text-xl md:text-2xl text-emerald-100 mb-10 max-w-3xl mx-auto leading-relaxed">
                        From docking results to publication-ready insights — in minutes.
                        <br className="hidden md:block" />
                        No scripting. No manual steps. Just science.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <a
                            href={downloadUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-3 bg-emerald-600 hover:bg-emerald-500 text-white px-8 py-4 rounded-xl font-semibold text-lg transition-all transform hover:scale-105 shadow-lg shadow-emerald-500/30 group"
                        >
                            <Download className="w-6 h-6 group-hover:translate-y-1 transition-transform" />
                            Download Desktop
                        </a>
                        <button
                            className="inline-flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white px-8 py-4 rounded-xl font-medium text-lg transition-all border border-slate-700"
                        >
                            <Globe className="w-5 h-5" />
                            Launch Web App
                        </button>
                    </div>
                </div>
            </section>

            {/* Value Proposition */}
            <section className="py-20 bg-gray-50">
                <div className="container mx-auto px-6">
                    <div className="text-center max-w-3xl mx-auto mb-16">
                        <h2 className="text-3xl font-bold mb-4 text-slate-900">Why BioDockViz?</h2>
                        <p className="text-gray-600 text-lg">
                            Modern docking tools generate enormous data—but interpreting it is slow. BioDockViz bridges the gap with an AI-assisted, seamless workflow.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        <FeatureCard
                            icon={<Zap className="w-8 h-8 text-emerald-600" />}
                            title="Instant Workflow"
                            desc="Upload → Analyze → Visualize → Export. One seamless process without complex installation or scripts."
                        />
                        <FeatureCard
                            icon={<CheckCircle className="w-8 h-8 text-emerald-600" />}
                            title="Scientifically Accurate"
                            desc="Rigorous validation of atom types, bond orders, and interaction thresholds. No silent errors."
                        />
                        <FeatureCard
                            icon={<FileText className="w-8 h-8 text-emerald-600" />}
                            title="Publication Ready"
                            desc="Export high-resolution (300 DPI) images with transparent backgrounds and automated captions."
                        />
                    </div>
                </div>
            </section>

            {/* Core Features Grid */}
            <section className="py-20">
                <div className="container mx-auto px-6">
                    <div className="grid md:grid-cols-2 gap-16 items-center">
                        <div className="order-2 md:order-1 bg-slate-900 rounded-2xl p-8 border border-slate-700 shadow-2xl relative overflow-hidden group">
                            <div className="absolute inset-0 bg-blue-500/10 blur-3xl group-hover:bg-blue-500/20 transition-all duration-700"></div>
                            {/* Mock UI for Visualization */}
                            <div className="relative bg-slate-800 rounded-lg p-4 border border-slate-700 h-80 flex items-center justify-center flex-col">
                                <Maximize className="w-16 h-16 text-slate-600 mb-4" />
                                <div className="text-slate-400 font-mono text-sm">3D Viewport</div>
                                <div className="absolute top-4 right-4 bg-black/50 px-3 py-1 rounded text-xs text-green-400 font-mono">
                                    H-Bond: 2.4Å
                                </div>
                                <div className="absolute bottom-4 left-4 flex gap-2">
                                    <div className="w-8 h-8 bg-slate-700 rounded flex items-center justify-center">📷</div>
                                    <div className="w-8 h-8 bg-slate-700 rounded flex items-center justify-center">⚙️</div>
                                </div>
                            </div>
                            <div className="mt-4 flex gap-3">
                                <div className="h-2 w-full bg-slate-700 rounded-full overflow-hidden">
                                    <div className="h-full bg-emerald-500 w-3/4"></div>
                                </div>
                                <span className="text-xs text-emerald-400 font-mono">Rendering...</span>
                            </div>
                        </div>

                        <div className="order-1 md:order-2">
                            <h2 className="text-3xl font-bold mb-6 text-slate-900">Advanced Interaction Analysis</h2>
                            <p className="text-gray-600 mb-8">
                                Automatically detect and visualize key interactions using scientifically accepted thresholds.
                            </p>
                            <ul className="space-y-6">
                                <DetailItem
                                    title="Multi-Format Support"
                                    desc="Seamlessly handle PDB, PDBQT, SDF, and MOL2 files. Multi-pose and multi-model support built-in."
                                />
                                <DetailItem
                                    title="AI-Assisted Insights"
                                    desc="Integrates AI reasoning to interpret binding modes and highlight key stabilizing interactions."
                                />
                                <DetailItem
                                    title="High-Performance 3D"
                                    desc="Smooth rendering of surfaces, cartoons, and sticks. Optimized for large protein-ligand complexes."
                                />
                                <DetailItem
                                    title="Intelligent Parsing"
                                    desc="Robust handling of edge cases and malformed files with correct atom/bond mapping."
                                />
                            </ul>
                        </div>
                    </div>
                </div>
            </section>

            {/* Target Audience */}
            <section className="py-20 bg-slate-50">
                <div className="container mx-auto px-6 text-center">
                    <h2 className="text-3xl font-bold mb-12 text-slate-900">Who Uses BioDockViz?</h2>
                    <div className="flex flex-wrap justify-center gap-6">
                        <AudienceBadge text="PhD Researchers" />
                        <AudienceBadge text="Pharmaceutical R&D" />
                        <AudienceBadge text="Academic Labs" />
                        <AudienceBadge text="Computational Chemists" />
                    </div>
                </div>
            </section>

            {/* Usage / CTA */}
            <section className="py-20 text-center">
                <div className="container mx-auto px-6">
                    <ShieldCheck className="w-16 h-16 text-emerald-600 mx-auto mb-6" />
                    <h2 className="text-3xl font-bold mb-6 text-slate-900">Start Visualizing in Seconds</h2>
                    <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto">
                        Secure, reliable, and ready for publication.
                    </p>
                    <a
                        href={downloadUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-3 bg-emerald-600 hover:bg-emerald-500 text-white px-10 py-5 rounded-xl font-bold text-xl shadow-xl transition-transform transform hover:scale-105"
                    >
                        <Download className="w-6 h-6" />
                        Download BioDockViz
                    </a>
                </div>
            </section>
        </div>
    );
};

// Sub-components
const FeatureCard = ({ icon, title, desc }) => (
    <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
        <div className="mb-6 bg-emerald-50 w-16 h-16 rounded-xl flex items-center justify-center">
            {icon}
        </div>
        <h3 className="text-xl font-bold mb-3 text-slate-900">{title}</h3>
        <p className="text-gray-600 leading-relaxed">{desc}</p>
    </div>
);

const DetailItem = ({ title, desc }) => (
    <li className="flex gap-4">
        <div className="mt-1">
            <CheckCircle className="w-6 h-6 text-emerald-500" />
        </div>
        <div>
            <h4 className="font-bold text-lg text-slate-900">{title}</h4>
            <p className="text-gray-600">{desc}</p>
        </div>
    </li>
);

const AudienceBadge = ({ text }) => (
    <span className="px-6 py-3 bg-white rounded-full border border-gray-200 text-gray-700 font-semibold shadow-sm">
        {text}
    </span>
);

export default BioDockVizPage;
