from setuptools import setup, find_packages
import os
import pkg_resources

setup(
  name='vingat',  # パッケージ名
  py_modules=["vingat"],
  version='0.1',  # バージョン
  packages=find_packages(),  # サブディレクトリに含まれるモジュールを自動で探す
  install_requires=[
    str(r)
    for r in pkg_resources.parse_requirements(
      open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    )
  ],
)
