import Link from 'next/link';
import { getSortedPostsData } from '@/lib/posts';
import { Cpu, Code, Terminal, Calendar, ArrowRight } from 'lucide-react';
import styles from './page.module.css';

export default function Home() {
  const allPostsData = getSortedPostsData();

  const getCategoryIcon = (cat) => {
    switch (cat) {
      case 'ai': return <Cpu size={14} />;
      case 'coding': return <Code size={14} />;
      case 'system': return <Terminal size={14} />;
      default: return <Code size={14} />;
    }
  };

  const getCategoryLabel = (cat) => {
    switch (cat) {
      case 'ai': return 'AI 人工智能';
      case 'coding': return 'Coding 编程';
      case 'system': return 'System 系统';
      default: return cat;
    }
  };

  return (
    <div>
      <section className={styles.hero}>
        <h1>👋 欢迎来到我的博客</h1>
        <p>探索 AI、编程技术和系统设计的世界。<br />在这里分享我的学习笔记、项目经验和技术思考。</p>

        <div className={styles.categories}>
          <Link href="/category/ai" className={`${styles.catBtn} ${styles.ai}`}>
            <Cpu size={18} /> AI
          </Link>
          <Link href="/category/coding" className={`${styles.catBtn} ${styles.coding}`}>
            <Code size={18} /> Coding
          </Link>
          <Link href="/category/system" className={`${styles.catBtn} ${styles.system}`}>
            <Terminal size={18} /> System
          </Link>
        </div>
      </section>

      <div className={styles.grid}>
        {allPostsData.map(({ id, date, title, category, excerpt }) => (
          <article key={id} className={styles.card}>
            <div className={styles.cardHeader}>
              <span className={`badge badge-${category}`}>
                {getCategoryIcon(category)} {category}
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
        ))}
      </div>
    </div>
  );
}
