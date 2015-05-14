{-|
Module      : Translation.Constants
Description : Define CUDA constants
Maintainer  : Josh Acay<cacay@cmu.edu>
Stability   : experimental
-}
module Translation.Constants (defineIndicies) where

import Control.Monad.State (State, execState, modify)
import qualified Data.Set as Set

import AST.AST
import AST.Operations


type Variables = Set.Set Ident

type Free = State Variables


defineIndicies :: Fun -> Fun
defineIndicies f@(Fun t id pars sts) =
  Fun t id pars (idx ++ idy ++ idz ++ sts)
  where
    define :: Ident -> Dimension -> Stmt
    define var d = Simp (Asgn (Ident var) $
      Binop Add
        (Binop Mul (CudaVar $ BlockIdx d) (CudaVar $ BlockDim d))
        (CudaVar $ ThreadIdx d))

    free = freeFun f
    idx = if Set.member "idx" free then [define "idx" DimX] else []
    idy = if Set.member "idy" free then [define "idy" DimY] else []
    idz = if Set.member "idz" free then [define "idz" DimZ] else []


freeFun :: Fun -> Variables
freeFun (Fun _ _ pars sts) = freeStmts Set.empty sts Set.\\ args
  where args = Set.fromList $ map (\(Param _ id) -> id) pars


freeStmts :: Variables -> [Stmt] -> Variables
freeStmts acc = foldr (flip freeStmt) acc


freeStmt :: Variables -> Stmt -> Variables
freeStmt acc st = case st of
  Simp s -> freeSimp acc s
  If e st1 st2 ->
    freeExp e `Set.union` freeStmts acc st1 `Set.union` freeStmts acc st2
  While e body -> freeExp e `Set.union` freeStmts acc body
  For init cond step body ->
    let header = freeExp cond `Set.union` freeSimp Set.empty step
    in freeSimp (header `Set.union` freeStmts acc body) init
  Break -> acc
  Continue -> acc
  Ret Nothing -> acc
  Ret (Just e) -> acc `Set.union` freeExp e


freeSimp :: Variables -> Simp -> Variables
freeSimp acc (Decl _ id) = Set.delete id acc
freeSimp acc (Asgn (Ident id) e) = freeExp e `Set.union` Set.delete id acc
freeSimp acc (Asgn e1 e2) = freeExp e1 `Set.union` freeExp e2 `Set.union` acc
freeSimp acc (Exp e) = acc `Set.union` freeExp e
freeSimp acc Nop = acc


freeExp :: Exp -> Variables
freeExp e = execState (freeExp' e) Set.empty
  where
    freeExp' :: Exp -> Free ()
    freeExp' Null = return ()
    freeExp' (Bool _) = return ()
    freeExp' (Int _) = return ()
    freeExp' (Float _) = return ()
    freeExp' (CudaVar v) = modify $ Set.insert $ show v
    freeExp' (Ident id) = modify $ Set.insert id
    freeExp' (Binop _ e1 e2) = freeExp' e1 >> freeExp' e2
    freeExp' (Cmp _ e1 e2) = freeExp' e1 >> freeExp' e2
    freeExp' (Case e1 e2 e3) = freeExp' e1 >> freeExp' e2 >> freeExp' e3
    freeExp' (Call _ args) = mapM_ freeExp' args
    freeExp' (Index e1 e2) = freeExp' e1 >> freeExp' e2
