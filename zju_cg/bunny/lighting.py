import numpy as np


def calculate_light_position(
    base_pos, animation_type, t, range_val, custom_params=None
):
    """
    计算特定时间点 (t=0..1) 的光源位置。

    Args:
        base_pos (np.ndarray): 光源的基础位置 (x, y, z).
        animation_type (str): 动画类型 ('none', 'vertical', 'custom', etc.).
        t (float): 归一化时间 (0.0 到 1.0).
        range_val (float): 动画的范围或幅度.
        custom_params (dict, optional): 用于 'custom' 动画的表达式字典
                                        {'x_expr': '...', 'y_expr': '...', 'z_expr': '...'}.

    Returns:
        np.ndarray: 计算得到的光源位置 (x, y, z).
    """
    pos = np.array(base_pos, dtype=np.float32)  # Start with base position
    pi = np.pi

    if animation_type == "none":
        return pos

    # Common trigonometric calculations
    sin_2pit = np.sin(2 * pi * t)
    cos_2pit = np.cos(2 * pi * t)

    # Apply animation based on type
    if animation_type == "vertical":
        pos[1] += range_val * sin_2pit
    elif animation_type == "horizontal":  # Move along X axis
        pos[0] += range_val * sin_2pit
    elif animation_type == "circular":  # Circle in XZ plane around base_pos
        pos[0] += range_val * sin_2pit
        pos[2] += range_val * cos_2pit
    elif animation_type == "pulse":  # Move along Z axis, simple in-out
        pos[2] += range_val * (1.0 - cos_2pit) / 2.0  # Range [0, range_val]
    elif animation_type == "figure8":  # Figure 8 in XY plane
        pos[0] += range_val * sin_2pit
        pos[1] += range_val * np.sin(4 * pi * t) / 2.0  # Double frequency for Y
    elif animation_type == "spiral":  # Spiral outwards in XY plane while moving along Z
        radius = range_val * t  # Radius increases with time
        angle_speed = 3  # Increase for faster spiral
        pos[0] += radius * np.sin(2 * pi * t * angle_speed)
        pos[1] += radius * np.cos(2 * pi * t * angle_speed)
        pos[2] += range_val * t  # Move along Z as well
    elif animation_type == "custom" and custom_params:
        # Create a safe evaluation environment
        # Include common math functions and constants
        safe_globals = {"__builtins__": {}}  # Restrict builtins
        safe_locals = {
            "t": t,
            "pi": pi,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "sqrt": np.sqrt,
            "exp": np.exp,
            "log": np.log,
            "abs": np.abs,
        }
        # Basic security check (very basic, not foolproof!)
        harmful_keywords = [
            "import",
            "exec",
            "eval",
            "open",
            "__",
            "lambda",
            "compile",
            "file",
            "input",
        ]

        for i, axis in enumerate(["x", "y", "z"]):
            expr_key = f"{axis}_expr"
            if expr_key in custom_params and custom_params[expr_key]:
                expr = custom_params[expr_key]
                # Check for harmful keywords
                if any(kw in expr for kw in harmful_keywords):
                    print(
                        f"错误: 在自定义表达式 '{expr_key}' 中检测到潜在不安全关键字: '{expr}'。跳过。"
                    )
                    continue
                try:
                    # Evaluate the expression safely
                    offset = range_val * eval(expr, safe_globals, safe_locals)
                    pos[i] += float(offset)  # Ensure result is float
                except NameError as e:
                    print(f"错误: 自定义表达式 '{expr_key}' 中存在未定义的名称: {e}")
                except SyntaxError as e:
                    print(f"错误: 自定义表达式 '{expr_key}' 中存在语法错误: {e}")
                except Exception as e:
                    print(f"错误: 评估自定义表达式 '{expr_key}' 时出错: {e}")
            # else: Keep base position for this axis if no expression provided
    else:
        print(
            f"警告: 未知的动画类型 '{animation_type}' 或缺少自定义参数。光源位置保持不变。"
        )

    return pos
