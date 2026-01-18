import Link from 'next/link';
import { getSortedPostsData } from '@/lib/posts';
import { Cpu, Code, Terminal, Calendar, ArrowRight } from 'lucide-react';
import styles from '@/app/page.module.css';

export function generateStaticParams() {
    return [
        { slug: 'ai' },
        { slug: 'coding' },
        { slug: 'system' },
    ];
}

export default async function Category({ params }) {
    const { slug } = await params;
    const allPostsData = getSortedPostsData();
    const categoryPosts = allPostsData.filter(post => post.category === slug);

    const getCategoryInfo = (cat) => {
        switch (cat) {
            case 'ai': return { icon: <Cpu size={48} />, title: 'AI 人工智能', desc: '探索大语言模型、机器学习、深度学习的最新技术和应用。' };
            case 'coding': return { icon: <Code size={48} />, title: 'Coding 编程技术', desc: '分享编程技术、代码实践、算法与数据结构。' };
            case 'system': return { icon: <Terminal size={48} />, title: 'System 系统设计', desc: '系统设计、架构模式、DevOps 实践和基础设施。' };
            default: return { icon: <Code size={48} />, title: cat, desc: '' };
        }
    };

    const info = getCategoryInfo(slug);

    return (
        <div>
            <div className={styles.hero} style={{ padding: '2rem 0' }}>
                <div style={{ color: `var(--${slug}-color)`, marginBottom: '1rem', display: 'flex', justifyContent: 'center' }}>
                    {info.icon}
                </div>
                <h1 style={{ color: `var(--${slug}-color)`, background: 'none', WebkitTextFillColor: 'initial' }}>
                    {info.title}
                </h1>
                <p>{info.desc}</p>
            </div>

            <div className={styles.grid}>
                {categoryPosts.length > 0 ? (
                    categoryPosts.map(({ id, date, title, category, excerpt }) => (
                        <article key={id} className={styles.card}>
                            <div className={styles.cardHeader}>
                                <span className={`badge badge-${category}`}>
                                    {category}
                                </span>
                                <span className={styles.date}>
                                    <Calendar size={12} /> {date}
                                </span>
                            </div>

                            <Link href={`/posts/${id}`}>
                                <h2 className={styles.cardTitle}>{title}</h2>
                            </Link>

                            <p className={styles.excerpt}>
                                {excerpt || '点击阅读更多内容...'}
                            </p>

                            <Link href={`/posts/${id}`} className={styles.readMore}>
                                阅读更多 <ArrowRight size={14} />
                            </Link>
                        </article>
                    ))
                ) : (
                    <p className="text-center col-span-full text-slate-500">该分类下暂无文章。</p>
                )}
            </div>
        </div>
    );
}
