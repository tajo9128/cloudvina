import { Link } from 'react-router-dom'

export default function BlogPage() {
    const posts = [
        {
            id: 1,
            title: 'Accelerating Drug Discovery with Cloud Computing',
            excerpt: 'How cloud-native molecular docking is changing the landscape of pharmaceutical research.',
            date: 'Nov 23, 2025',
            author: 'Dr. Sarah Chen'
        },
        {
            id: 2,
            title: 'Understanding AutoDock Vina Scoring Functions',
            excerpt: 'A deep dive into how Vina calculates binding affinity and what it means for your results.',
            date: 'Nov 20, 2025',
            author: 'James Wilson'
        },
        {
            id: 3,
            title: 'New Feature: SDF to PDBQT Converter',
            excerpt: 'We have just released a free tool to help you prepare your ligands for docking.',
            date: 'Nov 15, 2025',
            author: 'CloudVina Team'
        }
    ]

    return (
        <div className="container mx-auto px-4 py-12">
            <div className="text-center mb-12">
                <h1 className="text-4xl font-bold text-gray-900 mb-4">CloudVina Blog</h1>
                <p className="text-gray-600 max-w-2xl mx-auto">
                    Insights, updates, and tutorials from the world of computational chemistry.
                </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                {posts.map(post => (
                    <article key={post.id} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition">
                        <div className="p-6">
                            <div className="text-sm text-purple-600 font-medium mb-2">{post.date}</div>
                            <h2 className="text-xl font-bold text-gray-900 mb-3 hover:text-purple-600 cursor-pointer">
                                {post.title}
                            </h2>
                            <p className="text-gray-600 mb-4">
                                {post.excerpt}
                            </p>
                            <div className="flex items-center justify-between">
                                <span className="text-sm text-gray-500">By {post.author}</span>
                                <span className="text-purple-600 font-medium cursor-pointer hover:underline">Read more →</span>
                            </div>
                        </div>
                    </article>
                ))}
            </div>
        </div>
    )
}
