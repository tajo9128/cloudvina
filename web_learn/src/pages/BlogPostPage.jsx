import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { Calendar, User, Clock, ArrowLeft, Tag } from 'lucide-react';
import blogPosts from '../data/blogData';

export default function BlogPostPage() {
    const { slug } = useParams();
    const post = blogPosts.find(p => p.slug === slug);

    if (!post) {
        return (
            <div className="min-h-screen bg-white flex flex-col">
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <h1 className="text-4xl font-bold text-slate-900 mb-4">Post Not Found</h1>
                        <Link to="/blog" className="text-primary-600 hover:text-primary-700">
                            ← Back to Blog
                        </Link>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-white flex flex-col">

            <article className="flex-1">
                {/* Header */}
                <div className="bg-slate-50 border-b border-slate-200">
                    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                        <Link
                            to="/blog"
                            className="inline-flex items-center gap-2 text-slate-600 hover:text-primary-600 mb-6 transition-colors"
                        >
                            <ArrowLeft className="w-4 h-4" />
                            Back to Blog
                        </Link>

                        <div className="flex items-center gap-2 text-sm text-primary-600 font-semibold mb-4">
                            <Tag className="w-4 h-4" />
                            {post.category}
                        </div>

                        <h1 className="text-4xl md:text-5xl font-bold font-display text-slate-900 mb-6">
                            {post.title}
                        </h1>

                        <div className="flex flex-wrap items-center gap-6 text-slate-600">
                            <div className="flex items-center gap-2">
                                <User className="w-5 h-5" />
                                <span>{post.author}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <Calendar className="w-5 h-5" />
                                <span>{new Date(post.date).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <Clock className="w-5 h-5" />
                                <span>{post.readTime}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Featured Image */}
                {post.image && (
                    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 -mt-8 mb-12">
                        <img
                            src={post.image}
                            alt={post.title}
                            className="w-full h-96 object-cover rounded-xl shadow-xl"
                        />
                    </div>
                )}

                {/* Content */}
                <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
                    <div className="prose prose-lg max-w-none">
                        <p className="text-xl text-slate-600 leading-relaxed mb-8">
                            {post.excerpt}
                        </p>

                        {/* Placeholder for full blog content */}
                        <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 mb-8">
                            <p className="text-slate-700 mb-4">
                                <strong>Note:</strong> This is the blog post preview. The full content from the markdown files
                                will be rendered here once we integrate the markdown parser.
                            </p>
                            <p className="text-sm text-slate-600">
                                For now, you can read the excerpt above. Full content integration coming soon!
                            </p>
                        </div>
                    </div>

                    {/* Tags */}
                    <div className="mt-12 pt-8 border-t border-slate-200">
                        <h3 className="text-sm font-semibold text-slate-900 mb-4">Tags:</h3>
                        <div className="flex flex-wrap gap-2">
                            {post.tags.map(tag => (
                                <span
                                    key={tag}
                                    className="px-3 py-1.5 bg-slate-100 text-slate-700 rounded-full text-sm hover:bg-slate-200 transition-colors cursor-pointer"
                                >
                                    {tag}
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* Author Box */}
                    <div className="mt-12 p-6 bg-slate-50 rounded-xl border border-slate-200">
                        <div className="flex items-start gap-4">
                            <div className="w-16 h-16 bg-primary-600 rounded-full flex items-center justify-center text-white font-bold text-xl flex-shrink-0">
                                {post.author.charAt(0)}
                            </div>
                            <div>
                                <h3 className="font-bold text-slate-900 mb-1">{post.author}</h3>
                                <p className="text-slate-600 text-sm">
                                    Computational chemistry expert specializing in drug discovery and molecular docking.
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Related Posts */}
                    <div className="mt-16">
                        <h2 className="text-2xl font-bold font-display text-slate-900 mb-6">Related Articles</h2>
                        <div className="grid md:grid-cols-2 gap-6">
                            {blogPosts
                                .filter(p => p.id !== post.id && p.category === post.category)
                                .slice(0, 2)
                                .map(relatedPost => (
                                    <Link
                                        key={relatedPost.id}
                                        to={`/blog/${relatedPost.slug}`}
                                        className="block group"
                                    >
                                        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden hover:shadow-lg transition-shadow">
                                            <img
                                                src={relatedPost.image}
                                                alt={relatedPost.title}
                                                className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300"
                                            />
                                            <div className="p-4">
                                                <h3 className="font-bold text-slate-900 group-hover:text-primary-600 transition-colors mb-2">
                                                    {relatedPost.title}
                                                </h3>
                                                <p className="text-sm text-slate-600 line-clamp-2">{relatedPost.excerpt}</p>
                                            </div>
                                        </div>
                                    </Link>
                                ))}
                        </div>
                    </div>
                </div>
            </article>
        </div>
    );
}
