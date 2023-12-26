from setuptools import setup, find_packages

setup(
    name='Trading-Bot',
    version='0.1.0',
    author='Louis Spencer',
    author_email='louis@projectposeidon.io',
    description='An automated trading bot for cryptocurrencies',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Project-Poseidon/trading_bot',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'redshift_connector',
        'statsmodels',
        'scipy',
        'seaborn',
        'matplotlib',
        'joblib',
        'psutil',
        'vectorbt',
        'numba',
        'boto3'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
