{-|
Module      : AST.Type
Description : C types
Maintainer  : Josh Acay <cacay@cmu.edu>
Stability   : experimental
-}
module AST.Types (Type (..)) where

import Control.Arrow (first)
import Data.List (stripPrefix)
import Text.PrettyPrint
import Text.PrettyPrint.HughesPJClass (Pretty (..), prettyShow)


data Type = TVoid
          | TBool
          | TInt
          | TFloat
          | TDouble
          | TDim3
          | TArray Type
          | TFunction Type [Type]
            deriving (Eq, Ord)


instance Pretty Type where
  pPrint TVoid = text "void"
  pPrint TBool = text "bool"
  pPrint TInt = text "int"
  pPrint TFloat = text "float"
  pPrint TDouble = text "double"
  pPrint TDim3 = text "dim3"
  pPrint (TArray t) = pPrint t <> text "*"
  pPrint (TFunction t args) =
    parens (hcat $ punctuate (text " * ") $ map pPrint args) <+> text "->" <+> pPrint t

  pPrintList l = parens . fsep . punctuate comma . map (pPrintPrec l 0)


instance Show Type where
  show = prettyShow


instance Read Type where
  readsPrec p s
    | Just rest <- stripPrefix "void"   s = [(TVoid,   rest)]
    | Just rest <- stripPrefix "bool"   s = [(TBool,   rest)]
    | Just rest <- stripPrefix "int"    s = [(TInt,    rest)]
    | Just rest <- stripPrefix "float"  s = [(TFloat,  rest)]
    | Just rest <- stripPrefix "double" s = [(TDouble, rest)]
    | Just rest <- stripPrefix "dim3"   s = [(TDim3,   rest)]
    | Just rest <- stripPrefix "*"      s = map (first TArray) (readsPrec p rest)
  readsPrec _ _ = []
