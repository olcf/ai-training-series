/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
\*---------------------------------------------------------------------------*/

#include "radexIO.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type>
void Foam::radexIO::put
(
    const radex::data::OutgoingHandle& handle,
    const Field<Type>& field
) const
{
    // The tensor layout relies on Type being exactly its components
    static_assert
    (
        sizeof(Type) == pTraits<Type>::nComponents * sizeof(scalar),
        "Field element type is not contiguous scalar components"
    );

    putTensor
    (
        handle,
        reinterpret_cast<const scalar*>(field.cdata()),
        field.size(),
        pTraits<Type>::nComponents
    );
}


template<class Type>
void Foam::radexIO::put
(
    const radex::data::OutgoingHandle& handle,
    const GeometricField<Type, fvPatchField, volMesh>& field
) const
{
    put(handle, field.primitiveField());
}


template<class Type>
bool Foam::radexIO::get
(
    const radex::data::IncomingHandle& handle,
    Field<Type>& field,
    bool wait,
    scalar timeout
) const
{
    static_assert
    (
        sizeof(Type) == pTraits<Type>::nComponents * sizeof(scalar),
        "Field element type is not contiguous scalar components"
    );

    const label nComponents = pTraits<Type>::nComponents;

    List<scalar> data;
    List<label> shape;

    if (!fetchTensor(handle, wait, timeout, data, shape))
    {
        return false;
    }

    if (!shapeMatches(shape, field.size(), nComponents, handle.key()))
    {
        return false;
    }

    if (data.size() != field.size() * nComponents)
    {
        WarningInFunction
            << "Tensor " << handle.key() << " holds " << data.size()
            << " elements; expected " << field.size() * nComponents
            << "; skipping" << nl;

        return false;
    }

    std::copy
    (
        data.begin(),
        data.end(),
        reinterpret_cast<scalar*>(field.data())
    );

    return true;
}


template<class Type>
bool Foam::radexIO::get
(
    const radex::data::IncomingHandle& handle,
    GeometricField<Type, fvPatchField, volMesh>& field,
    bool wait,
    scalar timeout
) const
{
    if (!get(handle, field.primitiveFieldRef(), wait, timeout))
    {
        return false;
    }

    field.correctBoundaryConditions();

    return true;
}


// ************************************************************************* //
