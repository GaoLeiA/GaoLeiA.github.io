import { getAllPostIds, getPostData } from '@/lib/posts';
import { Calendar, ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export async function generateStaticParams() {
    const paths = getAllPostIds();
    return paths.map(path => path.params);
}

export async function generateMetadata({ params }) {
    const postData = await getPostData(params.slug);
    return {
        title: postData.title,
        description: postData.excerpt,
    };
}

export default async function Post({ params }) {
    const postData = await getPostData(params.slug);

    return (
        <div className="max-w-3xl mx-auto">
            <Link href="/" className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-8 transition-colors">
                <ArrowLeft size={16} /> 返回首页
            </Link>

            <article>
                <header className="mb-8 text-center">
                    <div className="flex justify-center gap-4 mb-6">
                        <span className={`badge badge-${postData.category || 'other'}`}>
                            {postData.category}
                        </span>
                        <span className="flex items-center gap-2 text-slate-400 text-sm">
                            <Calendar size={14} /> {postData.date}
                        </span>
                    </div>

                    <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-indigo-200 to-purple-200">
                        {postData.title}
                    </h1>
                </header>

                <div
                    className="prose prose-invert prose-lg max-w-none"
                    dangerouslySetInnerHTML={{ __html: postData.contentHtml }}
                />
            </article>

            <div className="mt-16 pt-8 border-t border-slate-800">
                <h3>相关文章</h3>
                {/* Potentially add related posts here later */}
            </div>
        </div>
    );
}
