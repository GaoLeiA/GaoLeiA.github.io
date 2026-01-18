import { Github, Mail } from 'lucide-react';

export const metadata = {
    title: '关于我 - Lei\'s Blog',
};

export default function About() {
    return (
        <div className="max-w-3xl mx-auto text-center">
            <div className="mb-8">
                <div className="text-6xl mb-4">👨‍💻</div>
                <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-purple-400 mb-4">
                    你好！我是 Lei
                </h1>
                <p className="text-xl text-slate-400">
                    一名热爱技术的软件工程师。
                </p>
            </div>

            <div className="bg-slate-800/50 p-8 rounded-2xl border border-slate-700 mb-12 text-left">
                <p className="mb-6 leading-relaxed text-slate-300">
                    在这个博客里，我会分享我的学习笔记、项目经验和技术思考。主要关注以下领域：
                </p>

                <div className="grid md:grid-cols-3 gap-6">
                    <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                        <h3 className="text-purple-400 font-bold mb-2">🤖 AI 人工智能</h3>
                        <p className="text-sm text-slate-400">探索大语言模型、机器学习、深度学习的最新进展。</p>
                    </div>
                    <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                        <h3 className="text-cyan-400 font-bold mb-2">💻 Coding 编程</h3>
                        <p className="text-sm text-slate-400">分享编程技巧、代码优化、算法与数据结构。</p>
                    </div>
                    <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                        <h3 className="text-amber-400 font-bold mb-2">⚙️ System 系统</h3>
                        <p className="text-sm text-slate-400">系统架构、分布式系统、性能优化等技术话题。</p>
                    </div>
                </div>
            </div>

            <div className="flex justify-center gap-6">
                <a
                    href="mailto:lei.gao@outlook.com"
                    className="flex items-center gap-2 px-6 py-3 rounded-full bg-slate-800 hover:bg-slate-700 text-white transition-all"
                >
                    <Mail size={20} /> lei.gao@outlook.com
                </a>
                <a
                    href="https://github.com/GaoLeiA"
                    target="_blank"
                    className="flex items-center gap-2 px-6 py-3 rounded-full bg-slate-800 hover:bg-slate-700 text-white transition-all"
                >
                    <Github size={20} /> GaoLeiA
                </a>
            </div>

            <blockquote className="mt-12 p-6 border-l-4 border-indigo-500 bg-indigo-500/10 italic text-indigo-200 rounded-r-lg inline-block">
                "技术是不断学习和探索的旅程，希望我的分享能对你有所帮助！ 🚀"
            </blockquote>
        </div>
    );
}
