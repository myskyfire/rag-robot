"""直接测试脚本 - 不使用交互模式"""
import sys
import os

# 添加项目根目录路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import RAGRobot

QUESTIONS = [
    # 一、基础文本问答
    "本标准规定了除什么以外的各种键的技术条件？",
    "本标准从什么时候开始实施？",
    "键的表面不允许有哪些缺陷？",
    "键的抗拉强度要求是多少？",
    # 二、条款细节问答
    "A型C型平键的圆弧部分有什么要求？",
    "半圆键两端圆角半径r的范围是多少？",
    "当键长L与键宽b之比大于等于8时，平行度按什么标准执行？",
    "楔键的斜度是多少？角度公差按哪一级？",
    # 三、表格数值问答
    "普通平键的键宽b对应的AQL值是多少？",
    "键高h的AQL值统一为多少？",
    "键长L的合格质量水平AQL是多少？",
    # 四、溯源与依据问答
    "请给出答案对应的标准条款号与页码。",
    "答案来自文档哪一段原文？",
    # 五、无答案问题
    "键的表面粗糙度要求是多少？",
    "平键的材料必须使用45号钢吗？",
    # 六、文本截断/残缺内容类
    "键长宽比大于等于8时，完整的平行度公差依据是什么？",
    "楔键极限偏差具体数值为多少？",
    # 七、模糊歧义提问
    "键外观上不能出现哪些不良形态？",
    "哪些类型的键需要管控圆弧偏斜问题？",
    # 八、表格交叉复合查询
    "薄型平键的键宽AQL合格水平是多少？",
    "普通平键、导向键、薄型平键三者键高的验收标准是否一致？",
    # 九、多条款组合问答
    "同时说明键力学强度与外观表面两项基本要求",
    "半圆键圆角取值大小和键的尺寸有什么关联",
    # 十、极值、范围类边界提问
    "半圆键允许的最小、最大圆角半径分别是多少",
    "本标准里验收等级数值最大的检查项目是哪一项",
    # 十一、同类概念对比提问
    "普通平键和导向键在键宽验收标准上有无区别",
    "楔键和平键在外形工艺要求上有什么不同规定",
    # 十二、文档完全无答案
    "键加工常用的淬火温度标准是多少",
    "存放键件的仓库温湿度要求是什么",
    "花键的尺寸公差参照本标准哪条规定",
    # 十三、条款编号精准检索
    "3.4条款对应的规范内容是什么",
    "4.2章节围绕什么检测指标展开",
    # 十四、语序颠倒、口语化非常规提问
    "2009年五月开始执行的这份标准，禁止键有什么瑕疵",
    "抗拉强度最低要达到多少才算合格键",
    # 十五、超文档业务延伸问题
    "按照这份标准生产的键，实际装配使用寿命大概多久",
    "不合格的键件官方允许返修处理吗",
]

def main():
    print("=" * 60)
    print("RAG Robot 批量测试")
    print("=" * 60)
    
    # 初始化机器人
    print("\n初始化RAG Robot...")
    robot = RAGRobot(collection_name="rag_robot_main")
    
    # 加载已有的向量库
    if robot.load():
        print("向量库加载成功")
    else:
        print("未找到已有索引，开始自动构建...")
        robot.scan_directory()
        robot.save()
        print("向量库构建完成")
    
    results = []
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"问题 {i}/{len(QUESTIONS)}: {question}")
        print(f"{'='*60}")
        
        try:
            answer = robot.ask(question)
            print(f"\n【回答】\n{answer}")
            results.append((question, answer, "成功"))
        except Exception as e:
            print(f"\n错误: {e}")
            results.append((question, str(e), "失败"))
    
    # 保存结果
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("RAG Robot 测试结果\n")
        f.write("=" * 60 + "\n\n")
        for i, (q, a, status) in enumerate(results, 1):
            f.write(f"问题 {i} [{status}]: {q}\n")
            f.write(f"回答:\n{a}\n")
            f.write("-" * 60 + "\n\n")
    
    print(f"\n{'='*60}")
    print(f"测试完成！")
    print(f"成功: {sum(1 for _, _, s in results if s == '成功')}")
    print(f"失败: {sum(1 for _, _, s in results if s == '失败')}")
    print(f"结果已保存到 test_results.txt")

if __name__ == "__main__":
    main()
