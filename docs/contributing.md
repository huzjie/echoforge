# 贡献指南

欢迎提交 PR。请遵循以下约定：

## 开发约定

1. **零依赖**：只用 Python 标准库，不引入 numpy/requests 等第三方依赖。
2. **确定性**：所有随机性走 `utils/rng.py`（`md5(key)` 前 8 字节种子），逐位可复现。
3. **闭式判真**：任何"对错"都必须能由 `core/oracle.py` 的纯函数精确判定。
4. **单调指标**：新增指标须随 `skill` 严格单调，且与其他指标互不重复。

## 检查清单

```bash
python -m compileall echoforge          # 零错误
python -m unittest discover -s tests -t .   # 全绿
python -m echoforge doctor              # 四项命中 1.0
python -m echoforge bench               # 单调 True
```

## 提交规范

- 一个 PR 一个主题；
- 新增能力需同步补 `tasks/registry.py`、`docs/`、`tests/`；
- README 基准数字必须真实测量，不手写估算。

## License

MIT。贡献即视为同意以 MIT 协议发布。
