{-|
Module      : AST
Description : Elaborated Python
Maintainer  : Josh Acay, Oguz Ulgen <cacay@cmu.edu, oulgen@andrew.cmu.edu>
Stability   : experimental

Python AST with fewer constructs and simplified expressions
-}
module AST.AST
  ( Ident
  , Fun (..)
  , Param (..)
  , Stmt (..)
  , Simp (..)
  , LValue
  , Exp (..)
  , CudaVar (..)
  , Dimension (..)
  , funName
  , funType
  ) where

import Data.List (stripPrefix)

import Text.PrettyPrint
import Text.PrettyPrint.HughesPJ (maybeParens)
import Text.PrettyPrint.HughesPJClass (Pretty (..), prettyShow)

import AST.Operations
import AST.Types (Type (..))


-- * Types

type Ident = String

data Fun = Fun Type Ident [Param] [Stmt]

data Param = Param Type Ident

data Stmt = Simp Simp
          | If Exp [Stmt] [Stmt]
          | While Exp [Stmt]
          | For Simp Exp Simp [Stmt]
          | Break
          | Continue
          | Ret (Maybe Exp)

data Simp = Decl Type Ident
          | Asgn LValue Exp
          | Exp Exp
          | Nop

type LValue = Exp

data Exp = Null
         | Bool Bool
         | Int Integer
         | Float Double
         | CudaVar CudaVar
         | Ident Ident
         | Binop Arithmetic Exp Exp
         | Cmp Comparison Exp Exp
         | Case Exp Exp Exp
         | Call Ident [Exp]
         | Index Exp Exp

data CudaVar = GridDim Dimension
             | BlockDim Dimension
             | BlockIdx Dimension
             | ThreadIdx Dimension
             | WarpSize

data Dimension = DimX | DimY | DimZ



-- * Analysis

funName :: Fun -> Ident
funName (Fun _ id _ _) = id


funType :: Fun -> Type
funType (Fun t id pars _) = TFunction t $ map (\(Param t _) -> t) pars



-- * Pretty printing

indentation :: Int
indentation = 2


instance Pretty Fun where
  pPrint (Fun t id pars body) =
    pPrint t <+> text id <+> pPrint pars $$ pPrint body

  pPrintList l = vcat . punctuate (text "\n") . map (pPrintPrec l 0)


instance Pretty Param where
  pPrint (Param t id) = pPrint t <+> text id

  pPrintList l = parens . fsep . punctuate comma . map (pPrintPrec l 0)


instance Pretty Stmt where
  pPrint (Simp s) = pPrint s <> semi
  pPrint (If e st1 []) = text "if" <+> parens (pPrint e) $+$ pPrint st1
  pPrint (If e st1 st2) = text "if" <+> parens (pPrint e) $+$ pPrint st1
    <+> text "else" $+$ pPrint st2
  pPrint (While e body) = text "while" <+> parens (pPrint e) $+$ pPrint body
  pPrint (For init cond step body) = text "for"
    <+> parens (pPrint init <> semi <+> pPrint cond <> semi <+> pPrint step)
    $+$ pPrint body
  pPrint (Ret Nothing) = text "return" <> semi
  pPrint (Ret (Just e)) = text "return" <+> pPrint e <> semi
  pPrint Continue = text "continue" <> semi
  pPrint Break = text "break" <> semi

  pPrintList l sts =
    lbrace $+$ nest indentation (vcat $ map (pPrintPrec l 0) sts) $+$ rbrace


instance Pretty Simp where
  pPrint (Decl t id) = pPrint t <+> text id
  pPrint (Asgn id e) = pPrint id <+> equals <+> pPrint e
  pPrint (Exp e) = pPrint e
  pPrint Nop = space


instance Pretty Exp where
  pPrintPrec l prec e = case e of
    Null -> text "null"
    Bool False -> text "false"
    Bool True -> text "true"
    Int x -> pPrint x
    Float f -> pPrint f
    CudaVar v -> pPrintPrec l prec v
    Ident id -> text id
    Binop op e1 e2 -> maybeParens (p <= prec) $
      pPrintPrec l p e1 <+> pPrint op <+> pPrintPrec l p e2
      where p = precedence op
    Cmp cmp e1 e2 -> maybeParens (p <= prec) $
      pPrintPrec l p e1 <+> pPrint cmp <+> pPrintPrec l p e2
      where p = precedence cmp
    Case c e1 e2 -> maybeParens (p <= prec) $ pPrintPrec l p c
      <+> char '?' <+> pPrintPrec l p e1 <+> colon <+> pPrintPrec l p e2
      where p = 0.5
    Call id es | Just f <- stripPrefix "math." id -> text f <> pPrint es
               | otherwise -> text id <> pPrint es
    Index e1 e2 -> maybeParens (p < prec) $
      pPrintPrec l p e1 <> brackets (pPrintPrec l p e2)
      where p = 12

  pPrintList l = parens . fsep . punctuate comma . map (pPrintPrec l 0)


instance Pretty CudaVar where
  pPrint (GridDim d) = text "gridDim" <> char '.' <> pPrint d
  pPrint (BlockDim d) = text "blockDim" <> char '.' <> pPrint d
  pPrint (BlockIdx d) = text "blockIdx" <> char '.' <> pPrint d
  pPrint (ThreadIdx d) = text "threadIdx" <> char '.' <> pPrint d
  pPrint WarpSize = text "warpSize"


instance Pretty Dimension where
  pPrint DimX = char 'x'
  pPrint DimY = char 'y'
  pPrint DimZ = char 'z'


-- * Showing

instance Show Fun where
  show = prettyShow


instance Show Exp where
  show = prettyShow


instance Show CudaVar where
  show = prettyShow
