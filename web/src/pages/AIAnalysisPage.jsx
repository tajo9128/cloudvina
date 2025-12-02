import { Helmet } from 'react-helmet-async'
import AIExplainer from '../components/AIExplainer'
import SEOHelmet from '../components/SEOHelmet'

export default function AIAnalysisPage() {
    const schemaMarkup = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": "AI Molecular Docking Explainer",
        "description": "AI-powered tool to analyze and explain AutoDock Vina docking results",
        "applicationCategory": "EducationalApplication",
        "offers": {
            "@type": "Offer",
            "price": "0",
            "priceCurrency": "USD"
        },
        "featureList": [
            "Upload AutoDock Vina results",
            "AI-powered explanations",
            "Educational content for students",
            "Free to use"
        ],
        "provider": {
            "@type": "Organization",
            "name": "BioDockify"
        }
    }

    return (
        <>
            {/* SEO Optimization */}
            <SEOHelmet
                title="AI Molecular Docking Explainer | Free AutoDock Vina Analysis Tool"
                description="Upload your AutoDock Vina docking results and get instant AI-powered explanations. Free molecular docking analysis tool powered by Grok AI. Perfect for students and researchers."
                keywords="molecular docking, AutoDock Vina, AI analysis, drug discovery, binding affinity explanation, docking results analyzer, ai explainer, computational chemistry"
                canonical="https://biodockify.com/ai-analysis"
                schema={schemaMarkup}
            />

            <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white pt-24 pb-16">
                <div className="container mx-auto px-4">
                    {/* Hero Section */}
                    <div className="max-w-4xl mx-auto text-center mb-12">
                        <div className="inline-block p-4 bg-purple-100 rounded-full mb-6">
                            <svg className="w-16 h-16 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                            </svg>
                        </div>
                        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
                            AI Molecular Docking Explainer
                        </h1>
                        <p className="text-xl text-slate-600 mb-8">
                            Upload your AutoDock Vina results and get instant, educational AI-powered explanations
                        </p>
                        <div className="flex flex-wrap gap-3 justify-center text-sm">
                            <span className="px-4 py-2 bg-green-100 text-green-800 rounded-full font-medium">✓ 100% Free</span>
                            <span className="px-4 py-2 bg-blue-100 text-blue-800 rounded-full font-medium">✓ Powered by Grok AI</span>
                            <span className="px-4 py-2 bg-purple-100 text-purple-800 rounded-full font-medium">✓ Educational</span>
                            <span className="px-4 py-2 bg-orange-100 text-orange-800 rounded-full font-medium">✓ No Signup Required</span>
                        </div>
                    </div>

                    {/* Main Content */}
                    <div className="max-w-6xl mx-auto">
                        <div className="grid md:grid-cols-3 gap-8 mb-12">
                            {/* How It Works */}
                            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                                    <span className="text-2xl">📤</span>
                                </div>
                                <h3 className="text-lg font-bold text-slate-900 mb-2">1. Upload Files</h3>
                                <p className="text-slate-600 text-sm">
                                    Drag and drop any files related to your molecular docking analysis - log files, PDBQT outputs, or other relevant data.
                                </p>
                            </div>

                            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                                    <span className="text-2xl">🤖</span>
                                </div>
                                <h3 className="text-lg font-bold text-slate-900 mb-2">2. AI Analysis</h3>
                                <p className="text-slate-600 text-sm">
                                    Our AI instantly analyzes your results and provides a summary of key metrics.
                                </p>
                            </div>

                            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                                    <span className="text-2xl">💬</span>
                                </div>
                                <h3 className="text-lg font-bold text-slate-900 mb-2">3. Ask Questions</h3>
                                <p className="text-slate-600 text-sm">
                                    Get educational explanations in plain language. Perfect for students and researchers.
                                </p>
                            </div>
                        </div>

                        {/* Why Use This Tool - SEO Content */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 mb-8">
                            <h2 className="text-2xl font-bold text-slate-900 mb-6">Why Use the AI Molecular Docking Explainer?</h2>

                            <div className="prose max-w-none text-slate-700 space-y-4">
                                <p>
                                    <strong>Understanding molecular docking results can be challenging</strong>, especially for students and early-career researchers. Our AI Molecular Docking Explainer transforms complex AutoDock Vina output into clear, educational explanations that anyone can understand. Whether you're analyzing binding affinities, RMSD values, or docking poses, our free AI-powered tool provides instant insights into your computational chemistry results.
                                </p>

                                <p>
                                    <strong>Perfect for pharmacy and chemistry students</strong>, this tool bridges the gap between raw docking data and practical drug discovery knowledge. Instead of spending hours deciphering log files and searching through textbooks, you can upload your results and receive immediate, context-aware explanations. The AI uses analogies like "lock-and-key" mechanisms and real-world examples to make complex concepts accessible, helping you learn faster and understand deeper.
                                </p>

                                <p>
                                    <strong>Researchers save valuable time</strong> with our AI explainer. When reviewing multiple docking experiments, you can quickly assess which protein-ligand interactions show promise. The tool explains not just what the numbers mean, but why they matter for drug design. Whether you're optimizing lead compounds or validating computational predictions, getting AI-powered interpretations helps you make informed decisions about which candidates to pursue further.
                                </p>

                                <p>
                                    <strong>Completely free and accessible</strong>, our AI Molecular Docking Explainer works with any AutoDock Vina output, regardless of where you ran your simulations. You don't need to create an account or install special software. Simply upload your log file, and within seconds, you'll receive a comprehensive analysis. The tool supports batch analysis, allowing you to compare multiple docking runs and understand trends across different ligands or binding sites.
                                </p>

                                <p>
                                    <strong>Educational by design</strong>, every explanation focuses on helping you learn. The AI doesn't just answer questions—it teaches fundamental concepts like hydrogen bonding, hydrophobic interactions, and binding thermodynamics. Students can use this tool for homework assignments, researchers can clarify results for lab meetings, and educators can demonstrate docking principles with real data. The interactive chat format encourages exploration and deeper understanding of molecular recognition principles.
                                </p>

                                <p>
                                    <strong>Powered by advanced Grok AI technology</strong>, our explainer provides scientifically accurate information while maintaining accessibility. It analyzes binding affinities, RMSD values, pose clustering, and energy ranges to give you a complete picture of your docking results. Whether you're asking "Is -8.5 kcal/mol a good binding affinity?" or "Why do I have multiple poses?", the AI provides detailed, relevant answers that enhance your understanding of computational drug discovery.
                                </p>
                            </div>
                        </div>

                        {/* AI Explainer Component */}
                        <div className="max-w-4xl mx-auto">
                            <AIExplainer />
                        </div>

                        {/* Features Grid */}
                        <div className="grid md:grid-cols-2 gap-6 mt-12">
                            <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-6 border border-blue-100">
                                <h3 className="text-lg font-bold text-slate-900 mb-3">🎓 For Students</h3>
                                <ul className="space-y-2 text-slate-700 text-sm">
                                    <li>• Learn molecular docking concepts</li>
                                    <li>• Get help with homework assignments</li>
                                    <li>• Understand binding affinity values</li>
                                    <li>• Free educational resource</li>
                                </ul>
                            </div>

                            <div className="bg-gradient-to-br from-green-50 to-blue-50 rounded-xl p-6 border border-green-100">
                                <h3 className="text-lg font-bold text-slate-900 mb-3">🔬 For Researchers</h3>
                                <ul className="space-y-2 text-slate-700 text-sm">
                                    <li>• Quick analysis of docking results</li>
                                    <li>• Interpret RMSD and energy values</li>
                                    <li>• Compare multiple binding modes</li>
                                    <li>• No software installation needed</li>
                                </ul>
                            </div>
                        </div>

                        {/* FAQ Section for SEO */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 mt-12">
                            <h2 className="text-2xl font-bold text-slate-900 mb-6">Frequently Asked Questions</h2>

                            <div className="space-y-6">
                                <div>
                                    <h3 className="font-bold text-slate-900 mb-2">What file formats are supported?</h3>
                                    <p className="text-slate-600">We support all file types! Upload AutoDock Vina log files (.txt, .log), PDBQT files (.pdbqt), or any other files related to your molecular docking analysis. The AI will analyze the content and provide explanations.</p>
                                </div>

                                <div>
                                    <h3 className="font-bold text-slate-900 mb-2">Is it really free?</h3>
                                    <p className="text-slate-600">Yes! Our AI Molecular Docking Explainer is completely free to use, with no hidden costs or subscription fees.</p>
                                </div>

                                <div>
                                    <h3 className="font-bold text-slate-900 mb-2">Do I need to create an account?</h3>
                                    <p className="text-slate-600">For basic AI analysis, you need to be logged in. However, registration is quick and free!</p>
                                </div>

                                <div>
                                    <h3 className="font-bold text-slate-900 mb-2">How accurate are the AI explanations?</h3>
                                    <p className="text-slate-600">Our AI is powered by Grok and trained on scientific literature. Explanations are educational and scientifically sound, perfect for learning and preliminary analysis.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
