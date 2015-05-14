{-|
Module      : Translation.TypeInference
Description : Type inference for expressions
Maintainer  : Josh Acay <cacay@cmu.edu>
Stability   : experimental
-}
module Translation.TypeInference (Context, Error, infer) where

import Control.Monad
import Control.Monad.Error (throwError)
import Control.Monad.Reader (ReaderT (runReaderT), asks, lift)
import Data.List (stripPrefix)
import qualified Data.Map.Strict as Map

import AST.AST
import AST.Operations
import AST.Types (Type (..))
import Util.Error (assertMsg, liftMaybe, msum1)


type Context = Map.Map Ident Type

type Error = Either String

type State = ReaderT Context Error


infer :: Context -> Exp -> Error Type
infer ctx e = runReaderT (inferExp e) ctx


inferExp :: Exp -> State Type
inferExp exp = case exp of
  Null -> return TVoid
  Bool _ -> return TBool
  Int _ -> return TInt
  Float _ -> return TFloat
  CudaVar _ -> return TInt
  Ident id ->
    asks (Map.lookup id) >>= liftMaybe ("undefined variable: " ++ id)
  Binop op e1 e2 -> do
    t1 <- inferExp e1
    t2 <- inferExp e2
    lift $ applyAny exp (inferArithmetic op) [t1, t2]
  Cmp op e1 e2 -> do
    t1 <- inferExp e1
    t2 <- inferExp e2
    lift $ applyAny exp (inferComparison op) [t1, t2]
  Case c e1 e2 -> do
    checkExp exp c TBool;
    t1 <- inferExp e1
    t2 <- inferExp e2
    lift $ unify exp t1 t2
  Call "len" [l] -> return TInt
  Call id args -> do
    t1 <- inferFun id
    t2 <- mapM inferExp args
    lift $ applyAny exp t1 t2
  Index e1 e2 -> do
    t1 <- inferExp e1
    t2 <- inferExp e2
    lift $ index exp t1 t2


checkExp :: Show a => a -> Exp -> Type -> State Type
checkExp err e t = do t' <- inferExp e; lift (unify err t' t)


inferFun :: Ident -> State [Type]
inferFun s | Just f <- stripPrefix "math." s = lift (inferMath f)
inferFun s | Right ts <- inferCast s = return ts
inferFun s = do t <- inferExp (Ident s); return [t]


inferMath :: Ident -> Error [Type]
inferMath s
  | isF2F s = return [TFunction TFloat [TFloat]]
  | isFF2F s = return [TFunction TFloat [TFloat, TFloat]]
  | isFFF2F s = return [TFunction TFloat [TFloat, TFloat, TFloat]]
  | isD2D s = return [TFunction TDouble [TDouble]]
  | isDD2D s = return [TFunction TDouble [TDouble, TDouble]]
  | isDDD2D s = return [TFunction TDouble [TDouble, TDouble, TDouble]]
  where
    isF2F = flip elem
      [ "aconsf", "aconshf", "asinf", "asinhf", "atanf", "atanhf", "cbrtf"
      , "ceilf", "cosf", "coshf", "cospif", "erfcf", "erfcinvf", "erfcxf"
      , "erff", "erfinvf", "exp10f", "exp2f", "expf", "expm1f", "fabsf"
      , "floorf", "j0f", "j1f", "lgammaf", "log10f", "log1pf", "log2f"
      , "logbf", "logf", "nearbyintf", "rcbrtf", "rintf", "roundf", "rsqrtf"
      , "sinf", "sinhf", "sinpif", "sqrtf", "tanf", "tanhf", "tgammaf"
      , "truncf", "y0f", "y1f"
      ]
    isFF2F = flip elem
      [ "atan2f", "copysignf", "fdimf", "fdividef", "fmaxf", "fminf"
      , "fmodf", "hypotf", "nextafterf", "powf", "remainderf"
      ]
    isFFF2F = flip elem ["fmaf"]
    isD2D = flip elem
      [ "acons", "aconsh", "asin", "asinh", "atan", "atanh", "cbrt"
      , "ceil", "cos", "cosh", "cospi", "erfc", "erfcinv", "erfcx"
      , "erf", "erfinv", "exp10", "exp2", "exp", "expm1", "fabs"
      , "floor", "j0", "j1", "lgamma", "log10", "log1p", "log2"
      , "logb", "log", "nearbyint", "rcbrt", "rint", "round", "rsqrt"
      , "sin", "sinh", "sinpi", "sqrt", "tan", "tanh", "tgamma"
      , "trunc", "y0", "y1"
      ]
    isDD2D = flip elem
      [ "atan2", "copysign", "fdim", "fdivide", "fmax", "fmin"
      , "fmod", "hypot", "nextafter", "pow", "remainder"
      ]
    isDDD2D = flip elem ["fma"]
-- Others here
inferMath s = throwError $ "Not a math library function: " ++ s


inferCast :: Ident -> Error [Type]
inferCast s = case s of
  "(int)" -> return [toInt TInt, toInt TFloat, toInt TDouble]
  "(float)" -> return [toFloat TInt, toFloat TFloat, toFloat TDouble]
  "(double)" -> return [toDouble TInt, toDouble TFloat, toDouble TDouble]
  _ -> throwError $ "Not a cast: " ++ s
  where
    toInt t = TFunction TInt [t]
    toFloat t = TFunction TFloat [t]
    toDouble t = TFunction TDouble [t]


inferArithmetic :: Arithmetic -> [Type]
inferArithmetic op = case op of
  Add -> numeric
  Sub -> numeric
  Mul -> numeric
  Div -> numeric
  Mod -> numeric
  Shl -> justInts
  Shr -> justInts
  And -> justInts
  Xor -> justInts
  Ior -> justInts
  where
    ints = TFunction TInt [TInt, TInt]
    floats = TFunction TFloat [TFloat, TFloat]
    doubles = TFunction TDouble [TDouble, TDouble]

    justInts = [ints]
    numeric = [ints, floats, doubles]


inferComparison :: Comparison -> [Type]
inferComparison op = case op of
  Eq -> polymorphic
  Neq -> polymorphic
  Less -> numeric
  LessEq -> numeric
  Greater -> numeric
  GreaterEq -> numeric
  where
    bools = TFunction TBool [TBool, TBool]
    ints = TFunction TBool [TInt, TInt]
    floats = TFunction TBool [TFloat, TFloat]
    doubles = TFunction TBool [TDouble, TDouble]

    numeric = [ints, floats, doubles]
    polymorphic = bools : numeric


unify :: Show a => a -> Type -> Type -> Error Type
unify _ t1 t2 | t1 == t2 = return t1
unify err t1 t2 = throwError $
  "cannot unify " ++ show t1 ++ " with " ++ show t2 ++ " in " ++ show err


apply :: Show a => a -> Type -> [Type] -> Error Type
apply err (TFunction r args) pars = do
  assertMsg ("incorrect arity in " ++ show err) (length args == length pars)
  zipWithM_ (unify err) args pars
  return r
apply err t _ = throwError $
  "expected function got " ++ show t ++ " in " ++ show err


applyAny :: Show a => a -> [Type] -> [Type] -> Error Type
applyAny err [] _ = throwError $ "function has no type: " ++ show err
applyAny err funs args = msum1 $ map (flip (apply err) args) funs


index :: Show a => a -> Type -> Type -> Error Type
index err (TArray t1) t2 = do unify err t2 TInt; return t1
index err t _ = throwError $
  "expected array got " ++ show t ++ " in " ++ show err
