import { CheckCircle, Zap, Shield, HelpCircle } from 'lucide-react'
import { Link } from 'react-router-dom'
import SEOHelmet from '../components/SEOHelmet'

export default function PricingPage() {
    const tiers = [
        {
            name: "Free Trial",
            price: "$0",
            period: "/7 days",
            description: "Perfect for students and testing the platform.",
            features: [
                "10 Docking Runs / day",
                "Basic AutoDock Vina",
                "Public Support",
                "1GB Storage"
            ],
            cta: "Start Free Trial",
            ctaLink: "/auth/signup",
            highlighted: false
        },
        {
            name: "Pro Research",
            price: "$49",
            period: "/month",
            description: "For individual researchers and PhD students.",
            features: [
                "Unlimited Docking Runs",
                "Priority GPU Queue",
                "MD Simulation (5ns/run)",
                "Private Support",
                "50GB Storage",
                "PDF Report Generation"
            ],
            cta: "Upgrade to Pro",
            ctaLink: "/auth/signup?plan=pro",
            highlighted: true
        },
        {
            name: "Enterprise",
            price: "Custom",
            period: "",
            description: "For biotechs requiring dedicated infrastructure.",
            features: [
                "Dedicated GPU Clusters",
                "Custom MD Protocols (>100ns)",
                "API Access",
                "SLA & Compliance",
                "SSO Integration",
                "On-Premise Deployment Option"
            ],
            cta: "Contact Sales",
            ctaLink: "/contact",
            highlighted: false
        }
    ]

    return (
        <div className="bg-slate-50 min-h-screen pt-32 pb-24">
            <SEOHelmet
                title="Pricing Plans | BioDockify"
                description="Flexible pricing for every stage of drug discovery. Free tier for students, Pro for researchers, and Enterprise for biotechs."
                keywords="drug discovery pricing, molecular docking cost, bioinformatics subscription"
            />

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center max-w-3xl mx-auto mb-16">
                    <h1 className="text-4xl font-bold text-slate-900 mb-6">Simple, Transparent Pricing</h1>
                    <p className="text-xl text-slate-600">
                        Choose the plan that best fits your research needs. No hidden fees.
                        Cancel anytime.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-24">
                    {tiers.map((tier, i) => (
                        <div key={i} className={`relative bg-white rounded-2xl shadow-lg p-8 border ${tier.highlighted ? 'border-primary-500 ring-4 ring-primary-500/10' : 'border-slate-100'}`}>
                            {tier.highlighted && (
                                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-primary-600 text-white px-4 py-1 rounded-full text-sm font-bold uppercase tracking-wide">
                                    Most Popular
                                </div>
                            )}
                            <h3 className="text-xl font-bold text-slate-900 mb-2">{tier.name}</h3>
                            <div className="flex items-baseline mb-4">
                                <span className="text-4xl font-bold text-slate-900">{tier.price}</span>
                                <span className="text-slate-500 ml-1">{tier.period}</span>
                            </div>
                            <p className="text-slate-600 mb-8 h-12">{tier.description}</p>

                            <ul className="space-y-4 mb-8">
                                {tier.features.map((feature, j) => (
                                    <li key={j} className="flex items-start">
                                        <CheckCircle className="text-emerald-500 mr-3 shrink-0" size={20} />
                                        <span className="text-slate-700 text-sm">{feature}</span>
                                    </li>
                                ))}
                            </ul>

                            <Link
                                to={tier.ctaLink}
                                className={`block w-full py-4 rounded-xl text-center font-bold transition-all ${tier.highlighted
                                        ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg hover:shadow-primary-600/30'
                                        : 'bg-slate-50 text-slate-900 hover:bg-slate-100 border border-slate-200'
                                    }`}
                            >
                                {tier.cta}
                            </Link>
                        </div>
                    ))}
                </div>

                {/* FAQ Section */}
                <div className="max-w-3xl mx-auto">
                    <h2 className="text-2xl font-bold text-slate-900 mb-8 text-center">Frequently Asked Questions</h2>
                    <div className="space-y-6">
                        {[
                            { q: "Can I cancel my subscription?", a: "Yes, you can cancel your subscription at any time. Your access will continue until the end of the billing period." },
                            { q: "Is there a student discount?", a: "The Free Trial is designed for students. However, we offer 50% off Pro plans for academic email addresses (.edu)." },
                            { q: "Do you offer refunds?", a: "We offer a 7-day money-back guarantee if you're not satisfied with the Pro plan capabilities." }
                        ].map((faq, i) => (
                            <div key={i} className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm">
                                <h4 className="font-bold text-slate-900 mb-2 flex items-center gap-2">
                                    <HelpCircle size={18} className="text-primary-500" /> {faq.q}
                                </h4>
                                <p className="text-slate-600 text-sm ml-6">{faq.a}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}
