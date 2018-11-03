# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:09:38 2018

@author: USER
"""

"""
scikit-learn/sklearn/base.py
Base class for all estimators in scikit-learn
"""

# class BaseEstimator 
# sklearn 中所有评估器的基类

#-----------------------------------------------------------------------

# inspect 用于检查实时对象
# inspect 提供函数用于检查实时对象的信息

# 包括：模块 modules，类 class，方法 methods，函数 functions
#       追溯 tracebacks，框架对象 frame objects，代码对象 code objects

# 主要是： type checking， 
#          getting source code，
#          inpsecting classes 和 
#          检查解释器堆栈 interpreter stack
#
# inpsect.signature 

# 1.函数注解
#   python3 的语法可以让定义函数的时候对参数以及返回值提供注解
#   def test(t:'test') -> 'test':
#       s = 'test: %s' %t
#       return 0
#   获得注解的方法 test.__annotations__
#   或者是用 signature inspect.signature(test)
#   得到  <Signature (t:'test') -> 'test'>

# 2.Python 中的 * 和 ** 的用法
#   函数传递的可变参数定义 *用于处理多个无名参数 postional arguments（一般由 args表示，但是并不是语法强制），
#                         **用于传递多个有定义参数 kyeword arguments （一般有 kwargs 表示，但是不是语法强制）
#   example：def test(*args，**kwrags)
#                print(args)
#                print(kwrags)
#                print('---------------')
#   test(1,None,'t') -> (1,None,'t') 
#                       {} 
#                       -------
#   发现，输入的无名参数均由 * 定义的 args 传递，传递的形式是一个 tuple
#   test(x = 1, y = 't') -> ()
#                           {'x':1,'y':'t'}
#                           --------
#  输入的有定义参数均由 ** 定义的 kwrags 传递，传递的形式是一个词典 （可以用来创建词典~）
#  test(1, x = 0, None, y = 't', 'test') 
#  False!
#  SyntaxError 语法错误，positional argument follows keyword argument
#  也就是说 kwargs 参数需要放置在 args 的后面。
#  这个是Python的语法要求的，函数中使用 * 和 ** 定义 args 和 kwrag 时就要求 * 在 ** 的前面
#  否则也会返回语法错误

from inspect import signature
from . import __version__
from collections import defaultdict
from sklearn_linear_model_base__pprint import _pprint
import warnings

class BaseEstimator(object):
    
    """Base class for all estimators in scikit-learn
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

#---------------------------------------------------------------------
# 什么是 python 中函数的装饰器？
# 简单的理解就是：给函数添加新的功能（装饰~），同时，不影响函数的结构，防止程序结构混乱
# 虽然说起来是装饰，但是实际上，功能实现的时候更类似把函数用新功能包裹（wrap）了起来
# 举例
    
# example: def test0():
#              print('test0')
    
# 我想给函数添加新的功能，比如，print('new')。可以写一个新的函数：
    
#          def test_add(func): # 没错，函数也是可以参数进行传递
#              print('new')
#              func()
# ---> new
#      test0
    
# 但是如果我想给所有的函数都添加这个新的功能呢？重写所有的函数会添加大量的重复代码
# 方法：写一个修饰器
    
#          def add(func): # 以一个函数为参数进行传递
#
#              def wrapper(): 
#                  print('new')
#                  return func()
#
#              return wrapper
# 
# 这个函数的结构：以一个函数作为参数。wrapper，先添加新的功能（print('new')），然后
# 执行 func()。最后返回的是一个函数
#
# test1 = add(test0)
# 执行 test1() --> new
#                  test0
    
# 相当于为 test0 添加新的功能，这就相当于一个修饰器。
# 而在 python 中，修饰器的执行可以用 @ 来进行
    
#      @add
#      def test0():
#          print('test0')
    
    @classmethod
    
    # 获得 estimator 的参数名称的一个方法
    def _get_param_names(cls):
        
        """Get parameter names for the estimator"""
        
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        
        #---------------------------------------------------------------------
        # cls, self, @staticmethod @classmethod
        # 
        # 1. self, cls 并不是关键词，而是惯例用法。但是最好不要改变
        #
        # 2. self 指向的是类的实例，而不是类本身
        
        #    example: class test(object):
        #                 def __init__(self,t0):
        #                     self.t0 = t0
        #    x = test('test') -> print(x) -> <__main__.x object at 0x000001D0276B7080>
        
        #    可见是类的实例而不是类本身
        
        # 3. @staticmethod 什么是 python 类的静态方法 
        
        #    class test(object):
        
        #         def __init__(self,t0):
        #             self.t0 = t0
        
        #         @staticmethod
        #         def test0(t1):
        #             print(t1)
        
        #         @classmethod
        #         def test1(cls,t2):
        #             t3 = '%s' %t2
        #             test2 = cls(t3)
        #             return test2
        
        #         def test3(self):
        #             test.test0(self.t0)
        #
        #     @staticmethod 类中的静态方法，就相当于函数。正式一点，即是与类交互，而不是与
        #     类的实例进行交互的方法。这个函数不但可以在外部使用，
        #     而且可以在类内部调用，如上例中，test3 method，就使用了静态方法 test0。
        #     需要注意的是，静态方法是类的方法，所以在内部调用的时候也是要用 test.test0
        #     的方式。
        #     外部使用包括，直接从类调用:  test.test0('test') -> test
        #                   从类的实例调用 x = test('test')
        #                                  x.test0('test') -> test
        #
        # 4. @ classmethod 什么是 python 类的类方法
        #    我的理解，类的 classmethod 方法，其作用相当于类的一个修饰器
        #    上面的注释中，装饰器，通过以函数 func 为传递参数，添加一些程序，为
        #    函数 func 添加了一些别的功能。而这里，classmethod，以类本身为传递
        #    参数，通过一些程序段，为类实例本身添加了一些别的功能
        #   
        #    如上个例子中，classmethod test1
        #    他传递的参数是 cls，是指向类本身的参数。另外一个参数 t2 是用来实例化的参数
        #    y = test.test1(2)
        #    test1 类方法，通过 t3 = '%s' %t2，可以识别数字的输入。
        #    y.test3() --> 2
        #    这样就是为类提供了新的方法
        #
        #    那么这样提供新方法的作用又在那里呢？
        #    这样的好处在于，重构类的时候，不需要再修改构造函数 __init__， 而是直接
        #    添加一个 classmethod 就可以了。相当于一个新的构造函数
        
        
        # 这里，先试图获得 deprecated_original 的属性值
        # 如果这个类没有给出这个值，那么返回这个类的构造函数 __init__
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        
        # 如果返回的确实这个类的构造函数，说明构造函数并没有 deprecated_original
        # 没有并要去自我检查，return []
        if init is object.__init__:
            # No explicit constructor to introspect
            return []
        
        # 使用 inspect.signature 检查构造函数中所有的属性
        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init) # signature 可以 inspect 函数。我尝试了类实例但是报错
                                         # 而 init 是类的构造函数
        
        #-------------------------------------------------------------
        # 获得构造函数中除了 self (p.name != 'self') 
        # 和 python 关键字 (p.kind != p.VAR_KEYWOD)                                 
        # Consider the constructor parameters excluding 'self'
        
        #----------------------------------
        # python 函数的参数类型
        
        # 1. VAR_POSITIONAL 类型，即上文提到的 postional arguments （*arg）
        #    为什么叫位置参数呢？
        #    因为没有关键字，只能通过参数在定义时的位置来传递。具体一点，在
        #    *arg 中，传入的参数是以一个 tuple 来传递的，tuple中参数的位置唯一
        #    确定参数
        
        # 2. VAR_KEYWORD 类型，即上文提到的 keyword arguments （**kwags）
        #    使用关键字来定义参数
        
        # 3. POSITIONAL_OR_KEYWORD 类型，最常用的类型，没有* 以及 ** 前缀
        #    顾名思义可以使用位置，或者是关键字来传参数调用
        #    def test(t):
        #        print(t)
        #    可以 test(2) test(t = 2)
        
        # 4. KEYWORD_ONLY 类型，顾名思义，只能通过关键字来传递参数
        #    这其实是被迫的~。比如，（*args，a = 1），在通过位置传递参数都已经被 *arg
        #    占用的情况下，只能再通过关键字来传递参数。。。
        
        # 5. VAR_KEYWORD deprcated version
        
        # 参数定义和组合的方式
        
#        POSITIONAL_OR_KEYWORD
#        VAR_POSITIONAL
#        KEYWORD_ONLY
#        VAR_KEYWORD
#        POSITIONAL_OR_KEYWORD，VAR_POSITIONAL
#        POSITIONAL_OR_KEYWORD，KEYWORD_ONLY
#        POSITIONAL_OR_KEYWORD，VAR_KEYWORD
#        VAR_POSITIONAL，KEYWORD_ONLY
#        VAR_POSITIONAL，VAR_KEYWORD
#        KEYWORD_ONLY，VAR_KEYWORD
#        POSITIONAL_OR_KEYWORD，VAR_POSITIONAL，KEYWORD_ONLY
#        VAR_POSITIONAL，KEYWORD_ONLY，VAR_KEYWORD
#        POSITIONAL_OR_KEYWORD，VAR_POSITIONAL，KEYWORD_ONLY，VAR_KEYWORD


        #------------------------------------------------
        # 如何判断参数的类型？
        # def test(t0, t1, t2):
        #     print(t0, t1, t2) # 显然这些参数都是 POSITIONAL_OR_KEYWORD 类型的参数 
        # s = inspect.siganature(test)
        # for p in s.parameters.values():
        #     print(p.POSITIONAL_OR_KEYWORD)
        #     print(p.VAR_KEYWORD)
        # 发现都会输出。这怎么判断？
        # for p in s.parameters.values():
        #     print(p.kind)
        # 发现输出的是 POSITIONAL_OR_KEYWORD，说明判断的没有错
        # 尝试用 == 来判断
        # for p in s.parameters.values():
        #     print(p.kind == p.POSITIONAL_OR_KEYWORD)
        # 输出 True
        # for p in s.parameters.values():
        #     print(p.kind == p.VAR_KEYWORD)
        # 输出 False
        # 说明可以这样进行参数类型的判断
        
        # 这里，剔除 self 和 VAR_KEYWORD 的参数。不过为什么提出关键字参数？
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        
        # 如果其中有 位置关键字，报错。为什么？实际用到的时候再注释吧
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # 返回排序后的参数
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    # 通过 _get_param_names 方法获得 estimator 的所有参数，将名字和值都存入一个字典
    def get_params(self, deep=True):
        
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        
        out = dict() # 初始化一个字典
        
        for key in self._get_param_names(): # 获得参数名称 str
            value = getattr(self, key, None) # 从 self 中获得这个名称的属性值
            
            if deep and hasattr(value, 'get_params'): # 怎么会拥有这个属性？ 
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
                
            out[key] = value # 将属性和值存入字典
        return out
    
    #-----------------------------------------------------------------------------
    # 对于输入的属性及其对应值组成的dict，使用 set_params 方法给 self 来赋值属性
    def set_params(self, **params):
        
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        
        #----------------------------------------------------------
        # 这句语法的意思
        # 对于 VAR_KEYWORD **kwrags 或者是 VAR_POSITIONAL *args
        # 如果没有输入对应的关键字参数或者是位置参数
        # not params = False
        if not params: 
            # Simple optimization to gain speed (inspect is slow)
            return self
        
        # 获得参数词典
        valid_params = self.get_params(deep=True)

        #---------------------------------------------------------
        # 什么是 defaultdict？先说一下它的用处吧。我个人感觉，这个
        # 的用处就是在元素查找及统计的时候可以自动设置一个默认值
        # 防止报错.
        
        nested_params = defaultdict(dict)  # grouped by prefix
        
        for key, value in params.items(): # 之前注释过，使用 items 和 iteritems 来对 dict 进行遍历
                                          # 因为 dict 是无序的
            
            #-----------------------------------------------------------------
            # str 的 partition 方法。按照制定的分隔符对 str 进行分割
            # 得到一个 有三个元素的 tuple。分别为 分隔符左边的 str， 分隔符，
            # 分割符右边的 str。实际测试的时候发现，不管 str 中有几个分割符，
            # 都只按照第一个分隔符进行分割，而不会处理后续的分隔符。如果没有
            # 分隔符，那么第一个元素就是 str，第二和第三都是空
            
            key, delim, sub_key = key.partition('__') # __ 对应 Python magic method
            
            # 如果不在合法列表中，报错
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            # 如果是 magic method，先存在 nested_params 中
            if delim: 
                nested_params[key][sub_key] = value
            
            # 如果不是，直接 setattr 进行传参, var_postional
            else:
                setattr(self, key, value)
                valid_params[key] = value
        
        # 这里的表述并没有看懂
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
    
    #------------------------------------------------------
    # print BaseEstimator
    # 原来 _pprint 是在这里用的，用来打印 cls 的属性参数
    
    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)
   
   #------------------------------------------------------
   # magic method 中的 getstate 和 setstate
   # 他们是对应 pickle 的方法
   # pickle dumps
   # 如果定义了 getsetate 方法，就按照这个方法来 pickle 属性值
   # unpickle loads
   # 如果定义了 setstate 方法，就按照这个方法来 unpickle 属性值
   #
   # example：   class foo(object):
   
#                    def __init__(self, val = 5):
#                        self.val = val
#                        
#                    def __getstate__(self):
#                        print('pickled')
#                        self.val = 2
#                        return self.__dict__
#                    
#                    def __setstate__(self, d):
#                        print('unpickled', d)
#                        self.__dict__ = d
#                        self.val = 3
   
   #  可以发现，pickle -  unpickle 之后
   #  f = foo() fpickle = pickle.dumps(f) funpickle = pickle.loads(f)
   #  funpickle.val = 3
   #  但是打印出来的值是 2
   #  说明 setstate 方法可以再度赋值
   
    def __getstate__(self):
        
        try:
            
            #-----------------------------------------------------------------
            # super 方法 - 类的继承
            
            # 基本用法 如果 class B(A) 即 B 继承 A类 （这里属于单继承）
            # 那么 B 继承了 A 的方法。但是如果想在 B 中使用 A 中的方法呢？
            # 使用 super 方法调用 parent class 的方法
            # super(B, self).method(...) python2
            # super().method(...) python3 
            
            # example: class test0(object):
            #              
            #              def __init__(self, t0, t1):
            #                  self.t0 = t0
            #                  self.t1 = t1
            #
            #              def puts(self):
            #                  print(self.t0 + self.t1)
            
            #          class test1(test0):
            
            #              def __init__(self, t0, t1):
            #                  self.t0 = t0
            #                  self.t1 = t1
            
            #              def add(self):
            #                  super(test1, self).puts()
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #  super 的用法充满了 bug！
            #  1. 使用 super 正确的 bound 到 parents 类
            #     super().method(....)
            #     super(test1, self).method(...)
            #     请务必使用这个表达方式，记住是 test1 而不是 test0
            #  2. bug 'method(..) takes 1 positional argument but 2 were given'
            #     出现这个的错误代码是 super(test1, self).puts(self)
            #     从函数的表达上，好像没有错。然而，并非如此。因为每个与类相关联的
            #     的方法调用都自动传递实参 self。因此调用方法的时候 self 已经默认传递了，
            #     所以才会有 2 were given 的错误。
            #  3. 实际上，如果我对例子进行实例化。x = test1(1,2)
            #     x.puts() --> 3 None
            #     最后一个 None 是怎么来的呢？因为 test0 方法中 puts 没有 return
            #     但是有一个默认的 return None 起了作用
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #
            # 这先尝试调用 BaseEstimator parent 类的 getstate 方法？
            state = super(BaseEstimator, self).__getstate__()
            
        except AttributeError:
            state = self.__dict__.copy() # 如果没有这个方法，就在 pickle 时调用 __dict__ 方法

        if type(self).__module__.startswith('sklearn.'): # __module__ 是包含该类定义的模块名 str
                                                         # startwith 是 str 的一个方法，用于检查
                                                         # str 是否以某一特定的字符串开头。返回 boolen
            return dict(state.items(), _sklearn_version=__version__) # pickle 用的 __getstate__ 和 __dict__
                                                                     # 返回的是一个dict
        else:
            return state

    def __setstate__(self, state):
        
        if type(self).__module__.startswith('sklearn.'): # 先检查 pickle 的时候是否正常，否则报错
            pickle_version = state.pop("_sklearn_version", "pre-0.18")
            
            if pickle_version != __version__:
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, __version__),
                    UserWarning)
        try:
            super(BaseEstimator, self).__setstate__(state) # 先尝试 BaseEstimator 的 parent 类 setstate 方法
                                                           
        except AttributeError:
            self.__dict__.update(state)  # 不行再调用默认的 __dict__ 方法。这是个好的思路，防止自己设置的 pickle 出问题