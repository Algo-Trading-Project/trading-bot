from setuptools import setup, find_packages

setup(
    name='Trading-Bot',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/Project-Poseidon/trading_bot',
    license='MIT',
    author='Louis Spencer',
    author_email='louisspencer87@gmail.comd',
    description="Repository for Project Poseidon's trading bot.",
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
        'numba'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)