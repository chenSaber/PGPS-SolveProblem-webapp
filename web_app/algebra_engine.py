# algebra_engine.py
# 核心功能：负责将模型输出的“抽象机器算子”翻译为“带有人类可读 LaTeX 公式”的可视化步骤。
import math
import re
import sympy as sp

# 🌐 翻译字典 (全量补全版)
OP_TRANS = {
    'Add': '加法求和', 'Sub': '减法运算', 'Mul': '乘法运算', 'Div': '除法运算',
    'Half': '折半计算', 'Square': '平方运算', 'Sqrt': '开根号运算', 'Get': '提取最终答案',
    'Sum': '连续求和', 'Multiple': '连续乘积', 'Equal': '等量代换(解方程)',
    'Proportion': '比例关系', 'Ratio': '比值计算',
    'Sin': '正弦计算(Sin)', 'Cos': '余弦计算(Cos)', 'Tan': '正切计算(Tan)',
    'Cot': '余切计算(Cot)', 'Sec': '正割计算(Sec)', 'Csc': '余割计算(Csc)',
    'Gsin': '直角三角形正弦(Sin)', 'Gcos': '直角三角形余弦(Cos)', 'Gtan': '直角三角形正切(Tan)',
    'Gougu': '勾股定理', 'Geo_Mean': '射影定理/几何平均', 'Sin_Law': '正弦定理', 'Cos_Law': '余弦定理', 'Median': '中点/中位线公式',
    'Triangle_Area': '三角形面积', 'Triangle_Area_Sine': '三角形面积(正弦公式)',
    'Para_Area': '平行四边形面积', 'Kite_Area': '菱形/对角线面积', 'Trapezoid_Area': '梯形面积',
    'Circle_Area': '圆的面积', 'Sector_Area': '扇形面积', 'Tria_BH_Area': '三角形面积(底乘高)',
    'Tria_SAS_Area': '三角形面积(两边夹角)', 'Rect_Area': '矩形面积', 'Rhom_Area': '菱形面积(基于对角线)',
    'Trap_Area': '梯形面积', 'Circle_R_Area': '圆/扇形面积(基于半径)', 'Circle_D_Area': '圆/扇形面积(基于直径)',
    'ArcSeg_Area': '弓形面积', 'RNgon_B_Area': '正多边形面积(基于边长)',
    'RNgon_L_Area': '正多边形面积(外接圆)', 'RNgon_H_Area': '正多边形面积(基于边心距)',
    'Circle_Perimeter': '圆的周长', 'Arc_Length': '圆弧弧长', 'Perimeter': '多边形周长',
    'PRK_Perim': '矩形/平行四边形周长', 'Circle_R_Circum': '圆周长/弧长(基于半径)',
    'Circle_D_Circum': '圆周长/弧长(基于直径)',
    'Similar_Triangle_1': '相似三角形比例', 'Similar_Triangle_2': '相似多重比例',
    'Angle_Add': '角度/线段相加', 'Angle_Sub': '角度/线段相减', 'Angle_Bisector': '角平分线/中点关系',
    'Vertical_Angle': '对顶角/相等关系', 'Parallel_Line_Angle': '平行线角度关系',
    'Supplementary_Angle': '互补角关系(和为180°)', 'Complementary_Angle': '互余角关系(和为90°)',
    'Triangle_Interior_Angle': '三角形内角和(180°)', 'Quad_Interior_Angle': '四边形内角和(360°)',
    'Polygon_Interior_Angle': '多边形内角和', 'Polygon_Exterior_Angle': '多边形外角和',
    'Iso_Tri_Ang': '等腰三角形角度', 'Chord2_Ang': '圆周角与弧度关系',
    'TanSec_Ang': '弦切角公式', 'Ngon_Angsum': '多边形内角和'
}


def fmt(val):
    try:
        val = float(val)
        return str(round(val, 3)) if val % 1 != 0 else str(int(val))
    except:
        return str(val)


def build_visualization_steps(sem_seqs, var_values, pred_action_list, final_ans):
    var_map = {}
    for i, val in enumerate(var_values):
        try:
            var_map[f"N{i}"] = float(val)
        except ValueError:
            var_map[f"N{i}"] = val

    val_to_name = {}
    for seq in sem_seqs:
        if "=" in seq:
            parts = seq.split("=")
            if len(parts) >= 2:
                val_str = parts[-1].strip()
                try:
                    val = float(val_str)
                    for p in parts[:-1]:
                        name = p.strip()
                        if name.isalpha():
                            val_to_name[val] = name
                            break
                except:
                    for p in parts[:-1]:
                        name = p.strip()
                        if name.isalpha():
                            val_to_name[val_str] = name
                            break

    sym_map = {}
    for i, val in enumerate(var_values):
        try:
            f_val = float(val)
            sym_map[f"N{i}"] = val_to_name.get(f_val, f"N{i}")
        except:
            sym_map[f"N{i}"] = val_to_name.get(val, val)

    # 1. 支持符号变量初始化的环境池
    val_env = {}
    for k, v in var_map.items():
        if isinstance(v, (int, float)):
            val_env[k] = float(v)
        else:
            try:
                s = str(v).replace('^', '**')
                s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)  # 3x -> 3*x
                s = re.sub(r'(\d)\s*\(', r'\1*(', s)  # 4(x-2) -> 4*(x-2)
                s = re.sub(r'\)\s*(\d)', r')*\1', s)  # (x+2)3 -> (x+2)*3
                s = re.sub(r'([a-zA-Z])\s*\(', r'\1*(', s)  # x(x+2) -> x*(x+2)
                s = re.sub(r'\)\s*\(', r')*(', s)  # (x+1)(x-1) -> (x+1)*(x-1)
                val_env[k] = sp.sympify(s)
            except:
                val_env[k] = v

    # 🌟 动态注册所有 C 系列常量 (如 C6, C90, C180 等)
    val_env['Cpi'] = sp.pi
    sym_map['Cpi'] = '\\pi'

    for token in pred_action_list:
        if isinstance(token, str) and re.match(r'^C\d+$', token):
            num_val = float(token[1:])
            val_env[token] = num_val
            if num_val in [90, 180, 360]:
                sym_map[token] = f"{int(num_val)}^\\circ"
            else:
                sym_map[token] = str(int(num_val))

    chunks = []
    current_chunk = []
    for token in pred_action_list:
        if not re.match(r'^([NV]\d+|[a-z]+|Ans|C\d+|Cpi)$', token):
            if current_chunk: chunks.append(current_chunk)
            current_chunk = [token]
        else:
            current_chunk.append(token)
    if current_chunk: chunks.append(current_chunk)

    steps = []

    for idx, chunk in enumerate(chunks):
        op = chunk[0]
        op_zh = OP_TRANS.get(op, op)
        chunk_str = " ".join(chunk)

        # 👇 ==================== 强化版：未知数判定雷达 ==================== 👇
        target_var = None
        for var in chunk[1:]:
            v = val_env.get(var)
            is_known = False

            if v is not None:
                if isinstance(v, (int, float)):
                    is_known = True
                elif isinstance(v, sp.Basic) and v.is_number:
                    is_known = True

            if not is_known:
                target_var = var
                break

        if not target_var:
            target_var = chunk[-1] if len(chunk) > 1 else f"V{idx}"
        # 👆 ================================================================ 👆

        if op == 'Get':
            res_v = chunk[1] if len(chunk) > 1 else 'Ans'
            sym_name = sym_map.get(res_v, res_v)
            action_display = f" 最终步骤：提取并输出计算结果 ({chunk_str})"
            equation_display = f"\\text{{最终结果}} \\; {sym_name} = {round(float(final_ans), 3) if final_ans is not None else '?'}"

            # 👇 修复：强制转换为 Python 原生 float，剥离 SymPy 属性
            safe_ans = float(final_ans) if final_ans is not None else None

            steps.append(
                {"step_num": idx + 1, "action": action_display, "equation": equation_display, "result": safe_ans})
            continue

        action_display = f"模型思考：应用【{op_zh}】求解未知量 ({chunk_str})"

        args = chunk[1:]
        cas_handled = False

        # 🚀 2. 终极 CAS 拦截层：统一所有高频代数算子与核心基础定理
        # (增加 Gougu, Geo_Mean, Median 进阶升舱！)
        if op in ['Sum', 'Equal', 'Proportion', 'Add', 'Sub', 'Mul', 'Div', 'Multiple', 'Gougu', 'Geo_Mean', 'Median']:
            try:
                equation = None
                if op == 'Sum' and len(args) >= 3:
                    lhs = sum(val_env.get(x, sp.Symbol(x)) for x in args[:-1])
                    rhs = val_env.get(args[-1], sp.Symbol(args[-1]))
                    equation = lhs - rhs
                elif op == 'Multiple' and len(args) >= 3:
                    lhs = val_env.get(args[0], sp.Symbol(args[0]))
                    for x in args[1:-1]:
                        lhs = lhs * val_env.get(x, sp.Symbol(x))
                    rhs = val_env.get(args[-1], sp.Symbol(args[-1]))
                    equation = lhs - rhs
                elif op == 'Equal' and len(args) == 2:
                    lhs = val_env.get(args[0], sp.Symbol(args[0]))
                    rhs = val_env.get(args[1], sp.Symbol(args[1]))
                    equation = lhs - rhs
                elif op == 'Proportion' and len(args) == 4:
                    lhs = val_env.get(args[0], sp.Symbol(args[0])) * val_env.get(args[3], sp.Symbol(args[3]))
                    rhs = val_env.get(args[1], sp.Symbol(args[1])) * val_env.get(args[2], sp.Symbol(args[2]))
                    equation = lhs - rhs
                elif op == 'Add' and len(args) == 3:
                    lhs = val_env.get(args[0], sp.Symbol(args[0])) + val_env.get(args[1], sp.Symbol(args[1]))
                    rhs = val_env.get(args[2], sp.Symbol(args[2]))
                    equation = lhs - rhs
                elif op == 'Sub' and len(args) == 3:
                    lhs = val_env.get(args[0], sp.Symbol(args[0])) - val_env.get(args[1], sp.Symbol(args[1]))
                    rhs = val_env.get(args[2], sp.Symbol(args[2]))
                    equation = lhs - rhs
                elif op == 'Mul' and len(args) == 3:
                    lhs = val_env.get(args[0], sp.Symbol(args[0])) * val_env.get(args[1], sp.Symbol(args[1]))
                    rhs = val_env.get(args[2], sp.Symbol(args[2]))
                    equation = lhs - rhs
                elif op == 'Div' and len(args) == 3:
                    lhs = val_env.get(args[0], sp.Symbol(args[0])) / val_env.get(args[1], sp.Symbol(args[1]))
                    rhs = val_env.get(args[2], sp.Symbol(args[2]))
                    equation = lhs - rhs
                elif op == 'Gougu' and len(args) == 3:
                    lhs = val_env.get(args[0], sp.Symbol(args[0]))**2 + val_env.get(args[1], sp.Symbol(args[1]))**2
                    rhs = val_env.get(args[2], sp.Symbol(args[2]))**2
                    equation = lhs - rhs
                elif op == 'Geo_Mean' and len(args) == 3:
                    lhs = val_env.get(args[2], sp.Symbol(args[2]))**2
                    rhs = val_env.get(args[0], sp.Symbol(args[0])) * val_env.get(args[1], sp.Symbol(args[1]))
                    equation = lhs - rhs
                elif op == 'Median' and len(args) == 3:
                    lhs = val_env.get(args[0], sp.Symbol(args[0])) + val_env.get(args[2], sp.Symbol(args[2]))
                    rhs = 2 * val_env.get(args[1], sp.Symbol(args[1]))
                    equation = lhs - rhs

                if equation is not None:
                    free_syms = list(equation.free_symbols)
                    target_sym = sp.Symbol(target_var)

                    solve_for = None
                    if target_sym in free_syms:
                        solve_for = target_sym
                    elif len(free_syms) == 1:
                        solve_for = free_syms[0]

                    if solve_for is not None:
                        sol = sp.solve(equation, solve_for)
                        if sol:
                            # 💡 核心修复：几何问题的多次幂可能存在负数解，这里安全地提取正数解
                            sol_val = sol[0]
                            for s in sol:
                                try:
                                    if float(s) > 0:
                                        sol_val = s
                                        break
                                except:
                                    pass

                            old_target_val = val_env.get(target_var)
                            val_env[str(solve_for)] = sol_val

                            if sol_val.is_number:
                                for k, v in val_env.items():
                                    if isinstance(v, sp.Basic) and solve_for in v.free_symbols:
                                        val_env[k] = v.subs(solve_for, sol_val)

                            def get_sym(k):
                                return sym_map.get(k, str(var_map.get(k, k)))

                            raw_lhs, raw_rhs = "", ""
                            if op == 'Sum':
                                raw_lhs = " + ".join([get_sym(x) for x in args[:-1]])
                                raw_rhs = get_sym(args[-1])
                            elif op == 'Multiple':
                                raw_lhs = " \\times ".join([get_sym(x) for x in args[:-1]])
                                raw_rhs = get_sym(args[-1])
                            elif op == 'Equal':
                                raw_lhs, raw_rhs = get_sym(args[0]), get_sym(args[1])
                            elif op == 'Proportion':
                                raw_lhs = f"\\frac{{{get_sym(args[0])}}}{{{get_sym(args[1])}}}"
                                raw_rhs = f"\\frac{{{get_sym(args[2])}}}{{{get_sym(args[3])}}}"
                            elif op == 'Add':
                                raw_lhs = f"{get_sym(args[0])} + {get_sym(args[1])}"; raw_rhs = get_sym(args[2])
                            elif op == 'Sub':
                                raw_lhs = f"{get_sym(args[0])} - {get_sym(args[1])}"; raw_rhs = get_sym(args[2])
                            elif op == 'Mul':
                                raw_lhs = f"{get_sym(args[0])} \\times {get_sym(args[1])}"; raw_rhs = get_sym(args[2])
                            elif op == 'Div':
                                raw_lhs = f"\\frac{{{get_sym(args[0])}}}{{{get_sym(args[1])}}}"; raw_rhs = get_sym(args[2])
                            elif op == 'Gougu':
                                raw_lhs = f"{get_sym(args[0])}^2 + {get_sym(args[1])}^2"; raw_rhs = f"{get_sym(args[2])}^2"
                            elif op == 'Geo_Mean':
                                raw_lhs = f"{get_sym(args[2])}^2"; raw_rhs = f"{get_sym(args[0])} \\times {get_sym(args[1])}"
                            elif op == 'Median':
                                raw_lhs = f"{get_sym(args[0])} + {get_sym(args[2])}"; raw_rhs = f"2 \\times {get_sym(args[1])}"

                            t_name = sym_map.get(str(solve_for), str(solve_for))

                            if not sol_val.is_number:
                                equation_display = f"{raw_lhs} = {raw_rhs} \\Rightarrow {t_name} = {sp.latex(sol_val)}"
                            else:
                                equation_display = f"{raw_lhs} = {raw_rhs} \\Rightarrow {t_name} = {fmt(float(sol_val))}"

                            if str(solve_for) != target_var:
                                new_t_val = val_env.get(target_var)
                                if isinstance(old_target_val, sp.Basic) and not old_target_val.is_number and new_t_val.is_number:
                                    t_sym_target = sym_map.get(target_var, target_var)
                                    equation_display += f" \\quad \\Rightarrow {t_sym_target} = {fmt(float(new_t_val))}"

                            steps.append({"step_num": idx + 1, "action": action_display, "equation": equation_display,
                                          "result": float(sol_val) if sol_val.is_number else None})
                            cas_handled = True
            except Exception as e:
                pass

        if cas_handled:
            continue

        # 🛡️ 3. 后备安全池：非 CAS 算子降级到浮点数执行，保障所有传统算子全部生效！
        try:
            v_args = []
            for x in args:
                v = val_env.get(x, 0.0)
                if isinstance(v, sp.Basic) and v.is_number:
                    v_args.append(float(v))
                elif isinstance(v, (int, float)):
                    v_args.append(float(v))
                else:
                    v_args.append(0.0)

            s_args = [sym_map.get(x, x) for x in args]
            eq_sym, eq_num, step_val = "", "", 0.0

            # ================ 基础代数 ================
            if op == 'Half' and len(args) == 2:
                if target_var == args[1]:
                    step_val = v_args[0] / 2; eq_sym = f"\\frac{{{s_args[0]}}}{{2}}"; eq_num = f"\\frac{{{fmt(v_args[0])}}}{{2}}"
                elif target_var == args[0]:
                    step_val = v_args[1] * 2; eq_sym = f"2 \\times {s_args[1]}"; eq_num = f"2 \\times {fmt(v_args[1])}"

            elif op == 'Square' and len(args) == 2:
                if target_var == args[1]:
                    step_val = v_args[0] ** 2; eq_sym = f"{s_args[0]}^2"; eq_num = f"{fmt(v_args[0])}^2"
                elif target_var == args[0]:
                    step_val = math.sqrt(max(0, v_args[1])); eq_sym = f"\\sqrt{{{s_args[1]}}}"; eq_num = f"\\sqrt{{{fmt(v_args[1])}}}"

            elif op == 'Sqrt' and len(args) == 2:
                if target_var == args[1]:
                    step_val = math.sqrt(max(0, v_args[0])); eq_sym = f"\\sqrt{{{s_args[0]}}}"; eq_num = f"\\sqrt{{{fmt(v_args[0])}}}"
                elif target_var == args[0]:
                    step_val = v_args[1] ** 2; eq_sym = f"{s_args[1]}^2"; eq_num = f"{fmt(v_args[1])}^2"

            # ================ 基础三角函数 ================
            elif op == 'Sin' and len(args) == 2:
                A, B = args[0], args[1]
                if target_var == B:
                    step_val = math.sin(math.radians(v_args[0])); eq_sym = f"\\sin({s_args[0]}^\\circ)"; eq_num = f"\\sin({fmt(v_args[0])}^\\circ)"
                elif target_var == A:
                    step_val = math.degrees(math.asin(max(-1.0, min(1.0, v_args[1])))); eq_sym = f"\\arcsin({s_args[1]})"; eq_num = f"\\arcsin({fmt(v_args[1])})"
            elif op == 'Cos' and len(args) == 2:
                A, B = args[0], args[1]
                if target_var == B:
                    step_val = math.cos(math.radians(v_args[0])); eq_sym = f"\\cos({s_args[0]}^\\circ)"; eq_num = f"\\cos({fmt(v_args[0])}^\\circ)"
                elif target_var == A:
                    step_val = math.degrees(math.acos(max(-1.0, min(1.0, v_args[1])))); eq_sym = f"\\arccos({s_args[1]})"; eq_num = f"\\arccos({fmt(v_args[1])})"
            elif op == 'Tan' and len(args) == 2:
                A, B = args[0], args[1]
                if target_var == B:
                    step_val = math.tan(math.radians(v_args[0])); eq_sym = f"\\tan({s_args[0]}^\\circ)"; eq_num = f"\\tan({fmt(v_args[0])}^\\circ)"
                elif target_var == A:
                    step_val = math.degrees(math.atan(v_args[1])); eq_sym = f"\\arctan({s_args[1]})"; eq_num = f"\\arctan({fmt(v_args[1])})"

            # ================ 直角三角函数 ================
            elif op == 'Gsin' and len(args) == 3:
                Opp, Hyp, Ang = args[0], args[1], args[2]
                if target_var == Opp:
                    step_val = v_args[1] * math.sin(math.radians(v_args[2])); eq_sym = f"{s_args[1]} \\times \\sin({s_args[2]}^\\circ)"; eq_num = f"{fmt(v_args[1])} \\times \\sin({fmt(v_args[2])}^\\circ)"
                elif target_var == Hyp:
                    step_val = v_args[0] / math.sin(math.radians(v_args[2])) if math.sin(math.radians(v_args[2])) != 0 else 0; eq_sym = f"\\frac{{{s_args[0]}}}{{\\sin({s_args[2]}^\\circ)}}"; eq_num = f"\\frac{{{fmt(v_args[0])}}}{{\\sin({fmt(v_args[2])}^\\circ)}}"
                elif target_var == Ang:
                    step_val = math.degrees(math.asin(max(-1.0, min(1.0, v_args[0] / v_args[1])))) if v_args[1] != 0 else 0; eq_sym = f"\\arcsin\\left(\\frac{{{s_args[0]}}}{{{s_args[1]}}}\\right)"; eq_num = f"\\arcsin\\left(\\frac{{{fmt(v_args[0])}}}{{{fmt(v_args[1])}}}\\right)"
            elif op == 'Gcos' and len(args) == 3:
                Adj, Hyp, Ang = args[0], args[1], args[2]
                if target_var == Adj:
                    step_val = v_args[1] * math.cos(math.radians(v_args[2])); eq_sym = f"{s_args[1]} \\times \\cos({s_args[2]}^\\circ)"; eq_num = f"{fmt(v_args[1])} \\times \\cos({fmt(v_args[2])}^\\circ)"
                elif target_var == Hyp:
                    step_val = v_args[0] / math.cos(math.radians(v_args[2])) if math.cos(math.radians(v_args[2])) != 0 else 0; eq_sym = f"\\frac{{{s_args[0]}}}{{\\cos({s_args[2]}^\\circ)}}"; eq_num = f"\\frac{{{fmt(v_args[0])}}}{{\\cos({fmt(v_args[2])}^\\circ)}}"
                elif target_var == Ang:
                    step_val = math.degrees(math.acos(max(-1.0, min(1.0, v_args[0] / v_args[1])))) if v_args[1] != 0 else 0; eq_sym = f"\\arccos\\left(\\frac{{{s_args[0]}}}{{{s_args[1]}}}\\right)"; eq_num = f"\\arccos\\left(\\frac{{{fmt(v_args[0])}}}{{{fmt(v_args[1])}}}\\right)"
            elif op == 'Gtan' and len(args) == 3:
                Opp, Adj, Ang = args[0], args[1], args[2]
                if target_var == Opp:
                    step_val = v_args[1] * math.tan(math.radians(v_args[2])); eq_sym = f"{s_args[1]} \\times \\tan({s_args[2]}^\\circ)"; eq_num = f"{fmt(v_args[1])} \\times \\tan({fmt(v_args[2])}^\\circ)"
                elif target_var == Adj:
                    step_val = v_args[0] / math.tan(math.radians(v_args[2])) if math.tan(math.radians(v_args[2])) != 0 else 0; eq_sym = f"\\frac{{{s_args[0]}}}{{\\tan({s_args[2]}^\\circ)}}"; eq_num = f"\\frac{{{fmt(v_args[0])}}}{{\\tan({fmt(v_args[2])}^\\circ)}}"
                elif target_var == Ang:
                    step_val = math.degrees(math.atan(v_args[0] / v_args[1])) if v_args[1] != 0 else 0; eq_sym = f"\\arcsin\\left(\\frac{{{s_args[0]}}}{{{s_args[1]}}}\\right)"; eq_num = f"\\arcsin\\left(\\frac{{{fmt(v_args[0])}}}{{{fmt(v_args[1])}}}\\right)"

            # ================ 角度拓扑定理 ================
            elif op == 'Iso_Tri_Ang' and len(args) == 2:
                V_ang, B_ang = args[0], args[1]
                if target_var == B_ang:
                    step_val = (180 - v_args[0]) / 2; eq_sym = f"\\frac{{180^\\circ - {s_args[0]}^\\circ}}{{2}}"; eq_num = f"\\frac{{180^\\circ - {fmt(v_args[0])}^\\circ}}{{2}}"
                elif target_var == V_ang:
                    step_val = 180 - 2 * v_args[1]; eq_sym = f"180^\\circ - 2 \\times {s_args[1]}^\\circ"; eq_num = f"180^\\circ - 2 \\times {fmt(v_args[1])}^\\circ"
            elif op == 'Supplementary_Angle' and len(args) == 2:
                if target_var == args[1]:
                    step_val = 180 - v_args[0]; eq_sym = f"180^\\circ - {s_args[0]}^\\circ"; eq_num = f"180^\\circ - {fmt(v_args[0])}^\\circ"
                elif target_var == args[0]:
                    step_val = 180 - v_args[1]; eq_sym = f"180^\\circ - {s_args[1]}^\\circ"; eq_num = f"180^\\circ - {fmt(v_args[1])}^\\circ"
            elif op == 'Complementary_Angle' and len(args) == 2:
                if target_var == args[1]:
                    step_val = 90 - v_args[0]; eq_sym = f"90^\\circ - {s_args[0]}^\\circ"; eq_num = f"90^\\circ - {fmt(v_args[0])}^\\circ"
                elif target_var == args[0]:
                    step_val = 90 - v_args[1]; eq_sym = f"90^\\circ - {s_args[1]}^\\circ"; eq_num = f"90^\\circ - {fmt(v_args[1])}^\\circ"
            elif op == 'Chord2_Ang' and len(args) == 3:
                A, B, C = args[0], args[1], args[2]
                if target_var == A: step_val = (v_args[1] + v_args[2]) / 2; eq_sym = f"\\frac{{{s_args[1]} + {s_args[2]}}}{{2}}"; eq_num = f"\\frac{{{fmt(v_args[1])} + {fmt(v_args[2])}}}{{2}}"
                elif target_var == B: step_val = 2 * v_args[0] - v_args[2]; eq_sym = f"2 \\times {s_args[0]} - {s_args[2]}"; eq_num = f"2 \\times {fmt(v_args[0])} - {fmt(v_args[2])}"
            elif op == 'TanSec_Ang' and len(args) == 3:
                A, B, C = args[0], args[1], args[2]
                if target_var == A: step_val = (v_args[2] - v_args[1]) / 2; eq_sym = f"\\frac{{{s_args[2]} - {s_args[1]}}}{{2}}"; eq_num = f"\\frac{{{fmt(v_args[2])} - {fmt(v_args[1])}}}{{2}}"
                elif target_var == C: step_val = 2 * v_args[0] + v_args[1]; eq_sym = f"2 \\times {s_args[0]} + {s_args[1]}"; eq_num = f"2 \\times {fmt(v_args[0])} + {fmt(v_args[1])}"

            # ================ 核心几何推导定理 ================
            elif op == 'Cos_Law' and len(args) == 4:
                SideA, SideB, SideC, AngA = args[0], args[1], args[2], args[3]
                if target_var == SideA:
                    step_val = math.sqrt(max(0, v_args[1] ** 2 + v_args[2] ** 2 - 2 * v_args[1] * v_args[2] * math.cos(math.radians(v_args[3]))))
                    eq_sym = f"\\sqrt{{{s_args[1]}^2 + {s_args[2]}^2 - 2 \\times {s_args[1]} \\times {s_args[2]} \\times \\cos({s_args[3]}^\\circ)}}"
                    eq_num = f"\\sqrt{{{fmt(v_args[1])}^2 + {fmt(v_args[2])}^2 - 2 \\times {fmt(v_args[1])} \\times {fmt(v_args[2])} \\times \\cos({fmt(v_args[3])}^\\circ)}}"
                elif target_var == AngA:
                    cos_val = (v_args[1] ** 2 + v_args[2] ** 2 - v_args[0] ** 2) / (2 * v_args[1] * v_args[2]) if (2 * v_args[1] * v_args[2]) != 0 else 0
                    step_val = math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))
                    eq_sym = f"\\arccos\\left(\\frac{{{s_args[1]}^2 + {s_args[2]}^2 - {s_args[0]}^2}}{{2 \\times {s_args[1]} \\times {s_args[2]}}}\\right)"
                    eq_num = f"\\arccos\\left(\\frac{{{fmt(v_args[1])}^2 + {fmt(v_args[2])}^2 - {fmt(v_args[0])}^2}}{{2 \\times {fmt(v_args[1])} \\times {fmt(v_args[2])}}}\\right)"
                else:
                    step_val = 0; eq_sym = "\\text{解一元二次方程}"; eq_num = "?"

            elif op == 'Sin_Law' and len(args) == 4:
                Ang1, Side1, Ang2, Side2 = args[0], args[1], args[2], args[3]
                if target_var == Side1:
                    step_val = v_args[3] * math.sin(math.radians(v_args[0])) / math.sin(math.radians(v_args[2])) if math.sin(math.radians(v_args[2])) != 0 else 0
                    eq_sym = f"\\frac{{{s_args[3]} \\times \\sin({s_args[0]}^\\circ)}}{{\\sin({s_args[2]}^\\circ)}}"; eq_num = f"\\frac{{{fmt(v_args[3])} \\times \\sin({fmt(v_args[0])}^\\circ)}}{{\\sin({fmt(v_args[2])}^\\circ)}}"
                elif target_var == Side2:
                    step_val = v_args[1] * math.sin(math.radians(v_args[2])) / math.sin(math.radians(v_args[0])) if math.sin(math.radians(v_args[0])) != 0 else 0
                    eq_sym = f"\\frac{{{s_args[1]} \\times \\sin({s_args[2]}^\\circ)}}{{\\sin({s_args[0]}^\\circ)}}"; eq_num = f"\\frac{{{fmt(v_args[1])} \\times \\sin({fmt(v_args[2])}^\\circ)}}{{\\sin({fmt(v_args[0])}^\\circ)}}"
            elif op == 'Ratio':
                if len(args) == 3:
                    A, B, C = args[0], args[1], args[2]
                    if target_var == C: step_val = v_args[0] / v_args[1] if v_args[1]!=0 else 0; eq_sym = f"\\frac{{{s_args[0]}}}{{{s_args[1]}}}"; eq_num = f"\\frac{{{fmt(v_args[0])}}}{{{fmt(v_args[1])}}}"
                    elif target_var == A: step_val = v_args[1] * v_args[2]; eq_sym = f"{s_args[1]} \\times {s_args[2]}"; eq_num = f"{fmt(v_args[1])} \\times {fmt(v_args[2])}"
                    elif target_var == B: step_val = v_args[0] / v_args[2] if v_args[2]!=0 else 0; eq_sym = f"\\frac{{{s_args[0]}}}{{{s_args[2]}}}"; eq_num = f"\\frac{{{fmt(v_args[0])}}}{{{fmt(v_args[2])}}}"
                elif len(args) == 4:
                    A, B, C, D = args[0], args[1], args[2], args[3]
                    if target_var == D: step_val = (v_args[0] / v_args[1]) ** v_args[2] if v_args[1]!=0 else 0; eq_sym = f"\\left(\\frac{{{s_args[0]}}}{{{s_args[1]}}}\\right)^{{{s_args[2]}}}"; eq_num = f"\\left(\\frac{{{fmt(v_args[0])}}}{{{fmt(v_args[1])}}}\\right)^{{{fmt(v_args[2])}}}"


            # ================ 面积周长计算 ================
            elif op == 'Triangle_Area' and len(args) == 3:
                if target_var == args[2]:
                        step_val = 0.5 * v_args[0] * v_args[1]; eq_sym = f"\\frac{{1}}{{2}} {s_args[0]} \\times {s_args[1]}"; eq_num = f"0.5 \\times {fmt(v_args[0])} \\times {fmt(v_args[1])}"
                elif target_var == args[0]:
                        step_val = (2 * v_args[2]) / v_args[1] if v_args[1] != 0 else 0; eq_sym = f"\\frac{{2 \\times {s_args[2]}}}{{{s_args[1]}}}"; eq_num = f"\\frac{{2 \\times {fmt(v_args[2])}}}{{{fmt(v_args[1])}}}"
                elif target_var == args[1]:
                        step_val = (2 * v_args[2]) / v_args[0] if v_args[0] != 0 else 0; eq_sym = f"\\frac{{2 \\times {s_args[2]}}}{{{s_args[0]}}}"; eq_num = f"\\frac{{2 \\times {fmt(v_args[2])}}}{{{fmt(v_args[0])}}}"
            elif op == 'Para_Area' and len(args) == 3:
                if target_var == args[2]:
                        step_val = v_args[0] * v_args[1]; eq_sym = f"{s_args[0]} \\times {s_args[1]}"; eq_num = f"{fmt(v_args[0])} \\times {fmt(v_args[1])}"
                elif target_var == args[0]:
                        step_val = v_args[2] / v_args[1] if v_args[1] != 0 else 0; eq_sym = f"\\frac{{{s_args[2]}}}{{{s_args[1]}}}"; eq_num = f"\\frac{{{fmt(v_args[2])}}}{{{fmt(v_args[1])}}}"
                elif target_var == args[1]:
                        step_val = v_args[2] / v_args[0] if v_args[0] != 0 else 0; eq_sym = f"\\frac{{{s_args[2]}}}{{{s_args[0]}}}"; eq_num = f"\\frac{{{fmt(v_args[2])}}}{{{fmt(v_args[0])}}}"
            elif op == 'Kite_Area' and len(args) == 3:
                if target_var == args[2]:
                        step_val = 0.5 * v_args[0] * v_args[1]; eq_sym = f"\\frac{{1}}{{2}} {s_args[0]} \\times {s_args[1]}"; eq_num = f"0.5 \\times {fmt(v_args[0])} \\times {fmt(v_args[1])}"
                elif target_var == args[0]:
                        step_val = (2 * v_args[2]) / v_args[1] if v_args[1] != 0 else 0; eq_sym = f"\\frac{{2 \\times {s_args[2]}}}{{{s_args[1]}}}"; eq_num = f"\\frac{{2 \\times {fmt(v_args[2])}}}{{{fmt(v_args[1])}}}"
                elif target_var == args[1]:
                        step_val = (2 * v_args[2]) / v_args[0] if v_args[0] != 0 else 0; eq_sym = f"\\frac{{2 \\times {s_args[2]}}}{{{s_args[0]}}}"; eq_num = f"\\frac{{2 \\times {fmt(v_args[2])}}}{{{fmt(v_args[0])}}}"
            elif op == 'Circle_Perimeter' and len(args) == 2:
                Radius, Perimeter = args[0], args[1]
                if target_var == Perimeter:
                    step_val = 2 * math.pi * v_args[0]; eq_sym = f"2 \\times \\pi \\times {s_args[0]}"; eq_num = f"2 \\times \\pi \\times {fmt(v_args[0])}"
                elif target_var == Radius:
                    step_val = v_args[1] / (2 * math.pi); eq_sym = f"\\frac{{{s_args[1]}}}{{2\\pi}}"; eq_num = f"\\frac{{{fmt(v_args[1])}}}{{2\\pi}}"
            elif op == 'Arc_Length' and len(args) == 3:
                Angle, Radius, Length = args[0], args[1], args[2]
                if target_var == Length:
                    step_val = (v_args[0] / 180) * math.pi * v_args[1]; eq_sym = f"\\frac{{{s_args[0]}^\\circ}}{{180^\\circ}} \\times \\pi \\times {s_args[1]}"; eq_num = f"\\frac{{{fmt(v_args[0])}^\\circ}}{{180^\\circ}} \\times \\pi \\times {fmt(v_args[1])}"
                elif target_var == Radius:
                    step_val = (v_args[2] * 180) / (v_args[0] * math.pi) if v_args[0] > 0 else 0; eq_sym = f"\\frac{{{s_args[2]} \\times 180^\\circ}}{{{s_args[0]}^\\circ \\times \\pi}}"; eq_num = f"\\frac{{{fmt(v_args[2])} \\times 180^\\circ}}{{{fmt(v_args[0])}^\\circ \\times \\pi}}"
                elif target_var == Angle:
                    step_val = (v_args[2] * 180) / (math.pi * v_args[1]) if v_args[1] > 0 else 0; eq_sym = f"\\frac{{{s_args[2]} \\times 180^\\circ}}{{\\pi \\times {s_args[1]}}}"; eq_num = f"\\frac{{{fmt(v_args[2])} \\times 180^\\circ}}{{\\pi \\times {fmt(v_args[1])}}}"
            elif op == 'Sector_Area' and len(args) == 3:
                Angle, Radius, Area = args[0], args[1], args[2]
                if target_var == Area:
                    step_val = (v_args[0] / 360) * math.pi * v_args[1] ** 2; eq_sym = f"\\frac{{{s_args[0]}^\\circ}}{{360^\\circ}} \\times \\pi \\times {s_args[1]}^2"; eq_num = f"\\frac{{{fmt(v_args[0])}^\\circ}}{{360^\\circ}} \\times \\pi \\times {fmt(v_args[1])}^2"
                elif target_var == Radius:
                    step_val = math.sqrt((v_args[2] * 360) / (v_args[0] * math.pi)) if v_args[0] > 0 and v_args[2] >= 0 else 0; eq_sym = f"\\sqrt{{\\frac{{{s_args[2]} \\times 360^\\circ}}{{{s_args[0]}^\\circ \\times \\pi}}}}"; eq_num = f"\\sqrt{{\\frac{{{fmt(v_args[2])} \\times 360^\\circ}}{{{fmt(v_args[0])}^\\circ \\times \\pi}}}}"
                elif target_var == Angle:
                    step_val = (v_args[2] * 360) / (math.pi * v_args[1] ** 2) if v_args[1] > 0 else 0; eq_sym = f"\\frac{{{s_args[2]} \\times 360^\\circ}}{{\\pi \\times {s_args[1]}^2}}"; eq_num = f"\\frac{{{fmt(v_args[2])} \\times 360^\\circ}}{{\\pi \\times {fmt(v_args[1])}^2}}"
            elif op == 'Tria_BH_Area' and len(args) == 3:
                A, B, C = args[0], args[1], args[2]
                if target_var == C: step_val = v_args[0] * v_args[1] / 2; eq_sym = f"\\frac{{1}}{{2}} \\times {s_args[0]} \\times {s_args[1]}"; eq_num = f"\\frac{{1}}{{2}} \\times {fmt(v_args[0])} \\times {fmt(v_args[1])}"
                elif target_var == A: step_val = 2 * v_args[2] / v_args[1] if v_args[1]!=0 else 0; eq_sym = f"\\frac{{2 \\times {s_args[2]}}}{{{s_args[1]}}}"; eq_num = f"\\frac{{2 \\times {fmt(v_args[2])}}}{{{fmt(v_args[1])}}}"
            elif op == 'Tria_SAS_Area' and len(args) == 4:
                A, B, C, D = args[0], args[1], args[2], args[3]
                if target_var == D: step_val = 0.5 * v_args[0] * v_args[2] * math.sin(math.radians(v_args[1])); eq_sym = f"\\frac{{1}}{{2}} {s_args[0]} \\times {s_args[2]} \\sin({s_args[1]}^\\circ)"; eq_num = f"0.5 \\times {fmt(v_args[0])} \\times {fmt(v_args[2])} \\sin({fmt(v_args[1])}^\\circ)"
            elif op == 'PRK_Perim' and len(args) == 3:
                A, B, C = args[0], args[1], args[2]
                if target_var == C: step_val = 2 * (v_args[0] + v_args[1]); eq_sym = f"2 \\times ({s_args[0]} + {s_args[1]})"; eq_num = f"2 \\times ({fmt(v_args[0])} + {fmt(v_args[1])})"
                elif target_var == A: step_val = v_args[2] / 2 - v_args[1]; eq_sym = f"\\frac{{{s_args[2]}}}{{2}} - {s_args[1]}"; eq_num = f"\\frac{{{fmt(v_args[2])}}}{{2}} - {fmt(v_args[1])}"
            elif op == 'Rect_Area' and len(args) == 3:
                A, B, C = args[0], args[1], args[2]
                if target_var == C: step_val = v_args[0] * v_args[1]; eq_sym = f"{s_args[0]} \\times {s_args[1]}"; eq_num = f"{fmt(v_args[0])} \\times {fmt(v_args[1])}"
                elif target_var == A: step_val = v_args[2] / v_args[1] if v_args[1]!=0 else 0; eq_sym = f"\\frac{{{s_args[2]}}}{{{s_args[1]}}}"; eq_num = f"\\frac{{{fmt(v_args[2])}}}{{{fmt(v_args[1])}}}"
            elif op == 'Rhom_Area' and len(args) == 3:
                A, B, C = args[0], args[1], args[2]
                if target_var == C: step_val = 2 * v_args[0] * v_args[1]; eq_sym = f"2 \\times {s_args[0]} \\times {s_args[1]}"; eq_num = f"2 \\times {fmt(v_args[0])} \\times {fmt(v_args[1])}"
                elif target_var == A: step_val = v_args[2] / (2 * v_args[1]) if v_args[1]!=0 else 0; eq_sym = f"\\frac{{{s_args[2]}}}{{2 \\times {s_args[1]}}}"; eq_num = f"\\frac{{{fmt(v_args[2])}}}{{2 \\times {fmt(v_args[1])}}}"
            elif op == 'Trap_Area' and len(args) == 4:
                A, B, C, D = args[0], args[1], args[2], args[3]
                if target_var == D: step_val = 0.5 * (v_args[0] + v_args[1]) * v_args[2]; eq_sym = f"\\frac{{({s_args[0]} + {s_args[1]}) \\times {s_args[2]}}}{{2}}"; eq_num = f"0.5 \\times ({fmt(v_args[0])} + {fmt(v_args[1])}) \\times {fmt(v_args[2])}"
                elif target_var == C: step_val = 2 * v_args[3] / (v_args[0] + v_args[1]) if (v_args[0]+v_args[1])!=0 else 0; eq_sym = f"\\frac{{2 \\times {s_args[3]}}}{{{s_args[0]} + {s_args[1]}}}"; eq_num = f"\\frac{{2 \\times {fmt(v_args[3])}}}{{{fmt(v_args[0])} + {fmt(v_args[1])}}}"
            elif op == 'Circle_R_Circum':
                if len(args) == 2:
                    R, C = args[0], args[1]
                    if target_var == C: step_val = 2 * math.pi * v_args[0]; eq_sym = f"2 \\pi {s_args[0]}"; eq_num = f"2 \\pi \\times {fmt(v_args[0])}"
                    elif target_var == R: step_val = v_args[1] / (2 * math.pi); eq_sym = f"\\frac{{{s_args[1]}}}{{2\\pi}}"; eq_num = f"\\frac{{{fmt(v_args[1])}}}{{2\\pi}}"
                elif len(args) == 3:
                    R, Ang, C = args[0], args[1], args[2]
                    if target_var == C: step_val = 2 * math.pi * v_args[0] * v_args[1] / 360; eq_sym = f"\\frac{{2\\pi \\times {s_args[0]} \\times {s_args[1]}^\\circ}}{{360^\\circ}}"; eq_num = f"\\frac{{2\\pi \\times {fmt(v_args[0])} \\times {fmt(v_args[1])}^\\circ}}{{360^\\circ}}"
            elif op == 'Circle_D_Circum':
                if len(args) == 2:
                    D, C = args[0], args[1]
                    if target_var == C: step_val = math.pi * v_args[0]; eq_sym = f"\\pi {s_args[0]}"; eq_num = f"\\pi \\times {fmt(v_args[0])}"
                    elif target_var == D: step_val = v_args[1] / math.pi; eq_sym = f"\\frac{{{s_args[1]}}}{{\\pi}}"; eq_num = f"\\frac{{{fmt(v_args[1])}}}{{\\pi}}"
            elif op == 'Circle_R_Area':
                if len(args) == 2:
                    R, A = args[0], args[1]
                    if target_var == A:
                        step_val = math.pi * v_args[0] ** 2; eq_sym = f"\\pi {s_args[0]}^2"; eq_num = f"\\pi \\times {fmt(v_args[0])}^2"
                    elif target_var == R:
                        step_val = math.sqrt(v_args[1] / math.pi) if v_args[1] >= 0 else 0; eq_sym = f"\\sqrt{{\\frac{{{s_args[1]}}}{{\\pi}}}}"; eq_num = f"\\sqrt{{\\frac{{{fmt(v_args[1])}}}{{\\pi}}}}"
                elif len(args) == 3:
                    R, Ang, A = args[0], args[1], args[2]
                    if target_var == A: step_val = math.pi * (v_args[0]**2) * v_args[1] / 360; eq_sym = f"\\frac{{\\pi {s_args[0]}^2 \\times {s_args[1]}^\\circ}}{{360^\\circ}}"; eq_num = f"\\frac{{\\pi \\times {fmt(v_args[0])}^2 \\times {fmt(v_args[1])}^\\circ}}{{360^\\circ}}"
            elif op == 'Circle_D_Area':
                if len(args) == 2:
                    D, A = args[0], args[1]
                    if target_var == A: step_val = 0.25 * math.pi * v_args[0]**2; eq_sym = f"\\frac{{\\pi {s_args[0]}^2}}{{4}}"; eq_num = f"\\frac{{\\pi \\times {fmt(v_args[0])}^2}}{{4}}"
            elif op == 'ArcSeg_Area' and len(args) == 3:
                R, Ang, A = args[0], args[1], args[2]
                if target_var == A: step_val = math.pi * v_args[0]**2 * v_args[1] / 360 - 0.5 * v_args[0]**2 * math.sin(math.radians(v_args[1])); eq_sym = f"\\frac{{\\pi {s_args[0]}^2 \\times {s_args[1]}^\\circ}}{{360^\\circ}} - \\frac{{1}}{{2}}{s_args[0]}^2 \\sin({s_args[1]}^\\circ)"; eq_num = f"\\frac{{\\pi {fmt(v_args[0])}^2 \\times {fmt(v_args[1])}^\\circ}}{{360^\\circ}} - 0.5 \\times {fmt(v_args[0])}^2 \\sin({fmt(v_args[1])}^\\circ)"
            elif op == 'Ngon_Angsum' and len(args) == 2:
                N, Sum = args[0], args[1]
                if target_var == Sum: step_val = (v_args[0] - 2) * 180; eq_sym = f"({s_args[0]} - 2) \\times 180^\\circ"; eq_num = f"({fmt(v_args[0])} - 2) \\times 180^\\circ"
                elif target_var == N: step_val = v_args[1] / 180 + 2; eq_sym = f"\\frac{{{s_args[1]}}}{{180^\\circ}} + 2"; eq_num = f"\\frac{{{fmt(v_args[1])}}}{{180^\\circ}} + 2"
            elif op == 'RNgon_B_Area' and len(args) == 3:
                N, Side, Area = args[0], args[1], args[2]
                if target_var == Area:
                    step_val = v_args[0] * v_args[1] ** 2 / (4 * math.tan(math.radians(180 / v_args[0]))) if v_args[0] != 0 else 0
                    eq_sym = f"\\frac{{{s_args[0]} \\times {s_args[1]}^2}}{{4 \\tan(\\frac{{180^\\circ}}{{{s_args[0]}}})}}"
                    eq_num = f"\\frac{{{fmt(v_args[0])} \\times {fmt(v_args[1])}^2}}{{4 \\tan(\\frac{{180^\\circ}}{{{fmt(v_args[0])}}})}}"
                elif target_var == Side:
                    step_val = math.sqrt((4 * v_args[2] * math.tan(math.radians(180 / v_args[0]))) / v_args[0]) if v_args[0] > 0 else 0
                    eq_sym = f"\\sqrt{{\\frac{{4 \\times {s_args[2]} \\times \\tan(\\frac{{180^\\circ}}{{{s_args[0]}}})}}{{{s_args[0]}}}}}"
                    eq_num = f"\\sqrt{{\\frac{{4 \\times {fmt(v_args[2])} \\times \\tan(\\frac{{180^\\circ}}{{{fmt(v_args[0])}}})}}{{{fmt(v_args[0])}}}}}"
            elif op == 'RNgon_L_Area' and len(args) == 3:
                N, Rad, Area = args[0], args[1], args[2]
                if target_var == Area:
                    step_val = 0.5 * v_args[0] * v_args[1] ** 2 * math.sin(math.radians(360 / v_args[0])) if v_args[0] != 0 else 0
                    eq_sym = f"\\frac{{1}}{{2}} {s_args[0]} {s_args[1]}^2 \\sin\\left(\\frac{{360^\\circ}}{{{s_args[0]}}}\\right)"
                    eq_num = f"0.5 \\times {fmt(v_args[0])} \\times {fmt(v_args[1])}^2 \\sin\\left(\\frac{{360^\\circ}}{{{fmt(v_args[0])}}}\\right)"
                elif target_var == Rad:
                    step_val = math.sqrt((2 * v_args[2]) / (v_args[0] * math.sin(math.radians(360 / v_args[0])))) if v_args[0] > 0 else 0
                    eq_sym = f"\\sqrt{{\\frac{{2 \\times {s_args[2]}}}{{{s_args[0]} \\sin(\\frac{{360^\\circ}}{{{s_args[0]}}})}}}}"
                    eq_num = f"\\sqrt{{\\frac{{2 \\times {fmt(v_args[2])}}}{{{fmt(v_args[0])} \\sin(\\frac{{360^\\circ}}{{{fmt(v_args[0])}}})}}}}"
            elif op == 'RNgon_H_Area' and len(args) == 3:
                N, Apothem, Area = args[0], args[1], args[2]
                if target_var == Area:
                    step_val = v_args[0] * v_args[1] ** 2 * math.tan(math.radians(180 / v_args[0])) if v_args[0] != 0 else 0
                    eq_sym = f"{s_args[0]} \\times {s_args[1]}^2 \\times \\tan\\left(\\frac{{180^\\circ}}{{{s_args[0]}}}\\right)"
                    eq_num = f"{fmt(v_args[0])} \\times {fmt(v_args[1])}^2 \\times \\tan\\left(\\frac{{180^\\circ}}{{{fmt(v_args[0])}}}\\right)"
                elif target_var == Apothem:
                    step_val = math.sqrt(v_args[2] / (v_args[0] * math.tan(math.radians(180 / v_args[0])))) if v_args[0] > 0 else 0
                    eq_sym = f"\\sqrt{{\\frac{{{s_args[2]}}}{{{s_args[0]} \\tan(\\frac{{180^\\circ}}{{{s_args[0]}}})}}}}"
                    eq_num = f"\\sqrt{{\\frac{{{fmt(v_args[2])}}}{{{fmt(v_args[0])} \\tan(\\frac{{180^\\circ}}{{{fmt(v_args[0])}}})}}}}"

            else:
                step_val = val_env.get(args[0], 0)
                eq_sym = f"\\text{{{op}}}({', '.join(s_args)})"
                eq_num = "?"

            if step_val is not None:
                val_env[target_var] = float(step_val)
            t_sym = sym_map.get(target_var, target_var)

            if eq_sym == eq_num:
                equation_display = f"{t_sym} = {eq_sym} = {round(float(step_val), 3)}"
            else:
                equation_display = f"{t_sym} = {eq_sym} = {eq_num} = {round(float(step_val), 3)}"

                # 👇 修复：强制转换为 Python 原生 float，剥离 SymPy 属性
            safe_step_val = float(step_val) if step_val is not None else None

            steps.append(
                {"step_num": idx + 1, "action": action_display, "equation": equation_display, "result": safe_step_val})

        except Exception as e:
            inputs = [x for x in chunk[1:] if x != target_var]
            in_syms = [sym_map.get(x, x) for x in inputs]
            equation_display = f"{sym_map.get(target_var, target_var)} = \\text{{{op}}}({', '.join(in_syms)}) = ?"
            steps.append({"step_num": idx + 1, "action": action_display, "equation": equation_display, "result": None})

    clean_var_map = {}
    for i, val in enumerate(var_values):
        n_key = f"N{i}"
        sym_name = sym_map.get(n_key, n_key)
        clean_var_map[f"{n_key} ({sym_name})"] = val

    raw_expression = " ".join(pred_action_list)
    return steps, clean_var_map, raw_expression