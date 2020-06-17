import numpy as np
import pandas as pd

import CyURT as urt
import statsmodels.api as sm


if __name__ == "__main__":

    x = np.array([[5, 15, 25, 35, 45, 55],[10,20,30,40,50,60]]).reshape((6, 2)).astype(np.float64)
    y = np.array([5, 20, 14, 32, 22, 38]).astype(np.float64)

    y = np.random.normal(size=(1000))
    x=np.random.normal(size=(1000,1))

    # yd = np.asarray(y).reshape(y.size)
    # yf = yd.astype(np.float32)
    # xd = np.asarray(x, order='F')
    # xf = xd.astype(np.float32)

    # running OLS regression as in ./examples/example1.cpp using double precision type
    fit = urt.OLS_d(y, sm.add_constant(x), True)
    fit.show()
    
    # running OLS regression as in ./examples/example1.cpp using single precision type
    fit = urt.OLS_f(y, x, True)
    fit.show()

    # running first ADF test as in ./examples/example2.cpp using double precision type
    test = urt.ADF_d(y, lags=None, trend=b'ct', method=b'AIC')
    test.show()

    # running second ADF test as in ./examples/example2.cpp using double precision type
    test.method = 'AIC'
    test.bootstrap = True
    test.niter = 10000
    test.show()

    model = sm.OLS(y, sm.add_constant(x))
    results = model.fit()
    results.params

    duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
    y = duncan_prestige.data['income']
    x = duncan_prestige.data['education']

    fit = urt.OLS_d(np.array(y).astype(np.float64), np.array(sm.add_constant(x)).astype(np.float64), True)
    fit.show()
    fit.param

    %%timeit
    test = urt.ADF_d(y, lags=None, trend=b'ct', method=b'AIC')

    %%timeit
    ts.adfuller(
                    y,
                    regression='ct',
                )