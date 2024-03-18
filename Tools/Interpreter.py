class Interpreter:
    def __init__(self):
        self.state = {}  # 用于存储变量状态
        self.output = []  # 用于存储输出结果

    def execute(self, code):
        try:
            # 在当前状态下执行代码
            exec(code, self.state)

            # 捕获代码输出
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                exec(code, self.state)

            output = f.getvalue()
            self.output.append(output)
        except Exception as e:
            self.output.append(str(e))
        return self.output[-1]

    def get_output(self):
        return '\n'.join(self.output)

    def clear_output(self):
        self.output = []
